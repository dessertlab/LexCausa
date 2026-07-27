#!/usr/bin/env bash
# =============================================================================
# run_ablation.sh — launcher for the taxonomy ablation (paper Appendix H).
#
# A paired A/B experiment (NOT part of the 1320-run campaign):
#   both roles = gpt_oss_120b; the thesis is generated once and frozen, and the
#   Counter-Reasoner runs twice per claim/replica:
#     Setup A (baseline):  counter_enable_causality = false  (no taxonomy)
#     Setup B (treatment): counter_enable_causality = true   (full taxonomy)
#   The only systematic difference is the taxonomic guidance to the opponent.
#
# End-to-end, single machine:
#   1. start Neo4j (single Docker container)
#   2. load / verify the knowledge base
#   3. start the Flask backend natively (must listen on the port used by the
#      endpoint in ablation_settings.json — 8000 by default)
#   4. run the A/B batch (scripts/run_doe_batch.py)
#   5. analyze the A/B results (scripts/analyze_doe_results.py)
#
# Prerequisites:
#   - conda env active:  conda activate lexcausa
#   - Docker running
#   - a filled .env (copy from .env.example): OPENROUTER_API_KEY, NEO4J_* ...
#
# Usage:
#   bash scripts/run_ablation.sh                 # full ablation + analysis
#   DRY_RUN=1 bash scripts/run_ablation.sh       # print the plan, run nothing
#   REPLICATES=2 bash scripts/run_ablation.sh    # fewer replicas (faster)
#   ONLY=C1,P2 bash scripts/run_ablation.sh      # restrict to some claims
#
# All-caps variables below are overridable from the environment.
# =============================================================================
set -euo pipefail

# ---- configuration (override via environment) --------------------------------
LLM_BACKEND="${LLM_BACKEND:-openrouter}"
CLAIMS="${CLAIMS:-claims.md}"
CONFIG="${CONFIG:-ablation_settings.json}"
REPLICATES="${REPLICATES:-5}"
DOMAINS="${DOMAINS:-CIVILE,PENALE,AMMINISTRATIVO,MIXED}"
ONLY="${ONLY:-}"                         # optional: C1,P2 to restrict the claim set
RUN_NAME="${RUN_NAME:-ablation_$(date +%Y%m%d_%H%M%S)}"

# infrastructure
API_PORT="${API_PORT:-8000}"             # MUST match the endpoint in ablation_settings.json
NEO4J_CONTAINER="${NEO4J_CONTAINER:-lexcausa-neo4j}"
NEO4J_IMAGE="${NEO4J_IMAGE:-neo4j:5.26}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-neo4jpassword}"
KEEP_BACKEND="${KEEP_BACKEND:-0}"        # 1 = leave the backend running afterwards
DRY_RUN="${DRY_RUN:-0}"

BATCH_RUNS_DIR="experiments/causal_taxonomy_ablation"
RUN_DIR="${BATCH_RUNS_DIR}/${RUN_NAME}"

# ---- resolve project root (this script lives in scripts/) --------------------
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
STARTED_BACKEND_PID=""

log() { printf '\n\033[1;36m[run_ablation]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[run_ablation] ABORT:\033[0m %s\n' "$*" >&2; exit 1; }

cleanup() {
    if [ "${KEEP_BACKEND}" != "1" ] && [ -n "${STARTED_BACKEND_PID}" ]; then
        log "stopping backend started by this script..."
        kill "${STARTED_BACKEND_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# --- 0. preflight -------------------------------------------------------------
log "[0/5] preflight"
command -v docker >/dev/null 2>&1 || die "docker not found"
command -v "$PYTHON" >/dev/null 2>&1 || die "python not found (did you 'conda activate lexcausa'?)"
[ -f "$CLAIMS" ]  || die "claims file not found: $CLAIMS"
[ -f "$CONFIG" ]  || die "config not found: $CONFIG"
[ -f .env ]       || die ".env not found — copy it from .env.example and fill in your keys"
if [ "$LLM_BACKEND" = "openrouter" ] && ! grep -qE '^\s*OPENROUTER_API_KEY=.+' .env; then
    die "OPENROUTER_API_KEY is not set in .env (required for LLM_BACKEND=openrouter)"
fi
log "run: $RUN_NAME  |  replicas: $REPLICATES  |  domains: $DOMAINS  |  backend: $LLM_BACKEND"
log "output will be: $RUN_DIR/"

if [ "$DRY_RUN" = "1" ]; then
    log "DRY_RUN=1 — showing the batch plan (no Neo4j/backend/LLM calls)."
    ONLY_ARG=(); [ -n "$ONLY" ] && ONLY_ARG=(--only "$ONLY")
    "$PYTHON" scripts/run_doe_batch.py \
        --config "$CONFIG" --claims-file "$CLAIMS" \
        --replicates "$REPLICATES" --domains "$DOMAINS" \
        --run-name "$RUN_NAME" "${ONLY_ARG[@]}" --dry-run
    exit 0
fi

# --- 1. Neo4j (single container) ----------------------------------------------
log "[1/5] Neo4j ($NEO4J_IMAGE, container '$NEO4J_CONTAINER')"
if docker ps --format '{{.Names}}' | grep -qx "$NEO4J_CONTAINER"; then
    log "Neo4j already running — reusing it."
elif docker ps -a --format '{{.Names}}' | grep -qx "$NEO4J_CONTAINER"; then
    log "starting existing Neo4j container..."
    docker start "$NEO4J_CONTAINER" >/dev/null
else
    log "creating Neo4j container..."
    docker run -d --name "$NEO4J_CONTAINER" \
        -p 7474:7474 -p 7687:7687 \
        -e NEO4J_AUTH="neo4j/${NEO4J_PASSWORD}" \
        -v lexcausa_neo4j_data:/data \
        "$NEO4J_IMAGE" >/dev/null
fi
log "waiting for Neo4j (bolt :7687)..."
until docker exec "$NEO4J_CONTAINER" cypher-shell -u neo4j -p "$NEO4J_PASSWORD" "RETURN 1;" >/dev/null 2>&1; do
    sleep 3
done
log "Neo4j is up."

# --- 2. knowledge base --------------------------------------------------------
log "[2/5] knowledge base"
if "$PYTHON" src/db/db_orchestrator.py --check 2>/dev/null | grep -qiE 'statute|precedent'; then
    log "KB already populated — reusing it (FORCE_RELOAD=1 to wipe & reload)."
    [ "${FORCE_RELOAD:-0}" = "1" ] && { log "FORCE_RELOAD=1 — clean reload..."; "$PYTHON" src/db/db_orchestrator.py --clean; }
else
    log "loading KB (statutes + precedents + embeddings + indexes)..."
    "$PYTHON" src/db/db_orchestrator.py
fi
"$PYTHON" src/db/db_orchestrator.py --check || die "KB verification failed"

# --- 3. backend ---------------------------------------------------------------
log "[3/5] backend on :$API_PORT (LLM_BACKEND=$LLM_BACKEND)"
if curl -sf "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
    log "backend already healthy on :$API_PORT — reusing."
else
    API_PORT="$API_PORT" LLM_BACKEND="$LLM_BACKEND" \
        "$PYTHON" src/api_server.py > "/tmp/lexcausa_backend_${API_PORT}.log" 2>&1 &
    STARTED_BACKEND_PID="$!"
    log "backend starting (log /tmp/lexcausa_backend_${API_PORT}.log)"
    until curl -sf "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; do sleep 2; done
    log "backend healthy (:$API_PORT)"
fi

# --- 4. A/B batch -------------------------------------------------------------
log "[4/5] running the A/B ablation batch"
ONLY_ARG=(); [ -n "$ONLY" ] && ONLY_ARG=(--only "$ONLY")
LLM_BACKEND="$LLM_BACKEND" "$PYTHON" scripts/run_doe_batch.py \
    --config "$CONFIG" --claims-file "$CLAIMS" \
    --replicates "$REPLICATES" --domains "$DOMAINS" \
    --run-name "$RUN_NAME" --resume "${ONLY_ARG[@]}"

SUMMARY_CSV="${RUN_DIR}/run_summary.csv"
[ -f "$SUMMARY_CSV" ] || die "batch finished but $SUMMARY_CSV not found — check the run logs"
log "batch complete. summary: $SUMMARY_CSV"

# --- 5. analysis --------------------------------------------------------------
log "[5/5] A/B analysis (paired t-test, sign test, Cohen's d, breakdowns)"
"$PYTHON" scripts/analyze_doe_results.py --csv "$SUMMARY_CSV"

log "DONE."
echo "  results  : $RUN_DIR/  (per-claim setup_A / setup_B / doe artifacts)"
echo "  summary  : $SUMMARY_CSV"
[ "$KEEP_BACKEND" = "1" ] && echo "  backend left running (KEEP_BACKEND=1)."
