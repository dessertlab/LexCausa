set -euo pipefail

# ---- configuration (override via environment) --------------------------------
LLM_BACKEND="${LLM_BACKEND:-openrouter}"
CLAIMS="${CLAIMS:-claims.md}"
R_MODELS="${R_MODELS:-gpt_oss_120b,llama_3_3_70b}"
C_MODELS="${C_MODELS:-gpt_oss_120b,llama_3_3_70b}"
PAIRING="${PAIRING:-cross}"              # cross = every R x C pairing (incl. self-play)
PLANNING="${PLANNING:-on,off,single}"   # the 3 paradigms: Plan-then-Execute / Step-wise / Single-call
REPLICATES="${REPLICATES:-5}"
SEED="${SEED:-42}"

# retrieval knobs — pinned to the main study; do not change to reproduce the paper
MIN_KEPT="${MIN_KEPT:-8}"
MAX_STATUTES="${MAX_STATUTES:-100}"
MAX_PRECEDENTS="${MAX_PRECEDENTS:-5}"

# infrastructure
WORKERS="${WORKERS:-1}"                  # parallel backends + DoE shards
BASE_PORT="${BASE_PORT:-8000}"          # worker i listens on BASE_PORT+i
NEO4J_CONTAINER="${NEO4J_CONTAINER:-lexcausa-neo4j}"
NEO4J_IMAGE="${NEO4J_IMAGE:-neo4j:5.26}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-neo4jpassword}"
KEEP_BACKEND="${KEEP_BACKEND:-0}"       # 1 = leave backends running after the run
DRY_RUN="${DRY_RUN:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-experiments/full_factorial/runs}"
ANALYSIS_OUT="${ANALYSIS_OUT:-experiments/full_factorial/analysis/main}"

# ---- resolve project root (this script lives in scripts/) --------------------
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
STARTED_BACKEND_PIDS=()

log() { printf '\n\033[1;36m[run_full]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[run_full] ABORT:\033[0m %s\n' "$*" >&2; exit 1; }

cleanup() {
    if [ "${KEEP_BACKEND}" != "1" ] && [ "${#STARTED_BACKEND_PIDS[@]}" -gt 0 ]; then
        log "stopping backend(s) started by this script..."
        for pid in "${STARTED_BACKEND_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    fi
}
trap cleanup EXIT

# --- 0. preflight -------------------------------------------------------------
log "[0/5] preflight"
command -v docker >/dev/null 2>&1 || die "docker not found"
command -v "$PYTHON" >/dev/null 2>&1 || die "python not found (did you 'conda activate lexcausa'?)"
[ -f "$CLAIMS" ] || die "claims file not found: $CLAIMS"
[ -f .env ] || die ".env not found — copy it from .env.example and fill in your keys"
if [ "$LLM_BACKEND" = "openrouter" ] && ! grep -qE '^\s*OPENROUTER_API_KEY=.+' .env; then
    die "OPENROUTER_API_KEY is not set in .env (required for LLM_BACKEND=openrouter)"
fi

# verify the run matrix with a dry run (no LLM calls)
log "verifying the run matrix (dry-run)..."
PREFLIGHT="$(LLM_BACKEND="$LLM_BACKEND" "$PYTHON" scripts/run_multi_doe.py \
    --claims-file "$CLAIMS" \
    --reasoner-models "$R_MODELS" --counter-models "$C_MODELS" \
    --pairing "$PAIRING" --planning-ablations "$PLANNING" \
    --replicates "$REPLICATES" --seed "$SEED" \
    --out /tmp/lexcausa_preflight --dry-run 2>&1)" || { echo "$PREFLIGHT"; die "dry-run failed"; }
echo "$PREFLIGHT" | sed 's/^/    /'
TOTAL="$(echo "$PREFLIGHT" | grep -oE 'Generated [0-9]+' | grep -oE '[0-9]+' | head -1 || echo 0)"
[ "${TOTAL:-0}" -gt 0 ] || die "run matrix is empty — check models/claims/paradigms"
log "run matrix: $TOTAL runs  |  workers: $WORKERS  |  backend: $LLM_BACKEND  |  out: $OUT"

if [ "$DRY_RUN" = "1" ]; then
    log "DRY_RUN=1 set — stopping before any real run."
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
    log "KB already populated — reusing it (use FORCE_RELOAD=1 to wipe & reload)."
    if [ "${FORCE_RELOAD:-0}" = "1" ]; then
        log "FORCE_RELOAD=1 — clean reload..."
        "$PYTHON" src/db/db_orchestrator.py --clean
    fi
else
    log "loading KB (statutes + precedents + embeddings + indexes)..."
    "$PYTHON" src/db/db_orchestrator.py
fi
"$PYTHON" src/db/db_orchestrator.py --check || die "KB verification failed"

# --- 3. backend(s) ------------------------------------------------------------
log "[3/5] backend(s) — $WORKERS worker(s), LLM_BACKEND=$LLM_BACKEND"
for i in $(seq 0 $((WORKERS-1))); do
    port=$((BASE_PORT+i))
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        log "worker $i: backend already healthy on :$port — reusing."
        continue
    fi
    API_PORT="$port" LLM_BACKEND="$LLM_BACKEND" \
        "$PYTHON" src/api_server.py > "/tmp/lexcausa_backend_${port}.log" 2>&1 &
    STARTED_BACKEND_PIDS+=("$!")
    log "worker $i: backend starting on :$port (log /tmp/lexcausa_backend_${port}.log)"
done
log "waiting for backend health..."
for i in $(seq 0 $((WORKERS-1))); do
    port=$((BASE_PORT+i))
    until curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; do sleep 2; done
    log "worker $i healthy (:$port)"
done

# --- 4. campaign --------------------------------------------------------------
log "[4/5] running the campaign ($TOTAL runs)"
COMMON_ARGS=(
    --claims-file "$CLAIMS"
    --reasoner-models "$R_MODELS" --counter-models "$C_MODELS"
    --pairing "$PAIRING" --planning-ablations "$PLANNING"
    --replicates "$REPLICATES" --seed "$SEED"
    --min-kept "$MIN_KEPT" --max-statutes "$MAX_STATUTES" --max-precedents "$MAX_PRECEDENTS"
)

if [ "$WORKERS" -eq 1 ]; then
    LLM_BACKEND="$LLM_BACKEND" "$PYTHON" scripts/run_multi_doe.py \
        "${COMMON_ARGS[@]}" \
        --api-url "http://localhost:${BASE_PORT}" \
        --shard-index 0 --shard-count 1 \
        --out "$OUT"
else
    log "sharding across $WORKERS workers..."
    pids=()
    for i in $(seq 0 $((WORKERS-1))); do
        port=$((BASE_PORT+i))
        LLM_BACKEND="$LLM_BACKEND" "$PYTHON" scripts/run_multi_doe.py \
            "${COMMON_ARGS[@]}" \
            --api-url "http://localhost:${port}" \
            --shard-index "$i" --shard-count "$WORKERS" \
            --out "${OUT}_g${i}" > "/tmp/lexcausa_doe_g${i}.log" 2>&1 &
        pids+=("$!")
        log "shard $i -> :$port (log /tmp/lexcausa_doe_g${i}.log)"
    done
    fail=0
    for idx in "${!pids[@]}"; do
        if wait "${pids[$idx]}"; then log "shard $idx OK"; else log "shard $idx FAILED (see its log)"; fail=1; fi
    done
    [ "$fail" -eq 0 ] || die "one or more shards failed"
    log "merging shards into $OUT ..."
    "$PYTHON" scripts/merge_doe_shards.py \
        --shards "${OUT}_g"* \
        --out "$OUT" --expect "$TOTAL"
fi
log "campaign complete. metrics: $OUT/metrics.csv"

# --- 5. analysis --------------------------------------------------------------
log "[5/5] statistical analysis (RQ1/RQ2/RQ3)"
"$PYTHON" scripts/analyze_multi_doe.py \
    --run-dir "$OUT" \
    --output "$ANALYSIS_OUT"

log "DONE."
echo "  runs     : $OUT/metrics.csv  (+ raw JSON in $OUT/runs/)"
echo "  analysis : $ANALYSIS_OUT/doe_analysis.json"
echo "  figures  : render experiments/full_factorial/analysis/lexcausa_stats.ipynb"
[ "$KEEP_BACKEND" = "1" ] && echo "  backends left running (KEEP_BACKEND=1)."
