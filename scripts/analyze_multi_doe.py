#!/usr/bin/env python
"""
Multi-DoE Analysis: statistical tests and reporting for the full-factorial
Reasoner x Counter-Reasoner DoE (HPC cluster metrics format).

Statistical methodology (see Appendix "Statistical Methods"; follows the
sampling-based testing procedure of DeepSample, Guerriero et al., ICSE 2024):

- Factorial ANOVA (kept): Reasoner, Counter, and planning main effects + the
  Reasoner x Counter interaction (RQ2), with the eta-squared importance index.
  Homoscedasticity is checked with Levene's test (normality via QQ plots).
- Friedman + Dunn/Holm (RQ1): for each multi-level factor (the 4 Reasoner and
  the 4 Counter models), the Friedman omnibus test blocked by the 22 claims;
  when it rejects, Dunn's post-hoc on the rank sums with Holm-Bonferroni
  correction, summarized as win/tie/loss matrices (rendered as heatmaps).
- Wilcoxon signed-rank: for the two-level contrasts blocked by claim, i.e. the
  model-class contrast (reasoning vs instruction-tuned, RQ1) and the planning
  ablation (Plan-then-Execute on/off, RQ3).
- Bootstrap CI for citation fidelity (a proportion).

The three AQA dimensions (Cogency, NormSupport, Semantics) are analyzed
separately when present, in addition to the aggregate net plausibility. Every
response is read on efficacy (quality) and efficiency (token cost).

NO paired t-test, Cohen's d, Kruskal-Wallis, or Mann-Whitney (removed: the data
is blocked by claim, so Friedman/Wilcoxon are the correct paired tests).

Expects metrics.csv (HPC cluster format) with columns including: reasoner_model,
counter_model, planning_reasoner, planning_counter, claim_id, aqa_plausibility,
citation_accuracy, citation_total, citation_repaired, total_tokens,
reasoning_tokens, counter_tokens, reasoning_steps, aqa_verdict, status, domain.
The AQA component columns (aqa_cogency, aqa_norm_support, aqa_semantics) are
analyzed if the run script exports them.

Usage:
    python scripts/analyze_multi_doe.py \\
        --run-dir experiments/full_factorial/runs/<ts> \\
        --output experiments/full_factorial/analysis/<ts>
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05

# Model aliases whose class is native reasoning (vs instruction-tuned).
REASONING_MODELS = {"deepseek_r1", "gpt_oss_120b"}

# Response variables. Quality (higher is better) and cost (lower is better).
# AQA components are analyzed only if exported by the run script.
QUALITY_RESPONSES = [
    "aqa_plausibility",
    "aqa_cogency",
    "aqa_norm_support",
    "aqa_semantics",
    "citation_accuracy",
]
COST_RESPONSES = ["total_tokens"]


def _model_class(alias: str) -> str:
    return "reasoning" if alias in REASONING_MODELS else "instruction_tuned"


class AdvancedDoEAnalyzer:
    """Statistical analysis for the Reasoner x Counter Multi-DoE."""

    def __init__(self, run_dir: str, output_dir: str):
        self.run_dir = Path(run_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = self.run_dir / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

        self.results_df = pd.read_csv(metrics_path)

        # Normalize the cloud single-axis format (one `model` used in both roles)
        # to the HPC cluster Reasoner x Counter format expected by the analysis below.
        if (
            "reasoner_model" not in self.results_df.columns
            and "model" in self.results_df.columns
        ):
            self.results_df["reasoner_model"] = self.results_df["model"]
            self.results_df["counter_model"] = self.results_df["model"]

        df = self.results_df
        if "status" in df.columns:
            df = df[df["status"] == "completed"]
        df = df.copy()
        # Planning is toggled jointly on both agents; use it as a single factor.
        if "planning_reasoner" in df.columns:
            df["planning"] = df["planning_reasoner"].astype(bool)
        if "reasoner_model" in df.columns:
            df["reasoner_class"] = df["reasoner_model"].apply(_model_class)

        # Reasoning-paradigm factor (3 levels), derived from the planning and
        # single-call flags (which are authoritative). This also backfills rows
        # whose `paradigm` column is missing or NaN, e.g. when a merged dataset
        # mixes older on/off runs (no paradigm column) with single-call runs.
        def _truthy(v: Any, default: bool) -> bool:
            if pd.isna(v):
                return default
            return str(v).strip().lower() in {"true", "1", "yes", "on", "t"}

        def _derive_paradigm(row: "pd.Series") -> str:
            if _truthy(row.get("single_call_reasoner"), False):
                return "single_call"
            return (
                "plan_then_execute"
                if _truthy(row.get("planning_reasoner"), True)
                else "stepwise"
            )

        # Canonical paradigm labels; a merged dataset can carry stray aliases
        # (e.g. "single") or blank cells (older on/off runs) that must be
        # normalized, otherwise the paradigm factor gets spurious extra levels
        # and the Friedman test fails.
        _PARADIGM_CANON = {
            "plan_then_execute": "plan_then_execute",
            "stepwise": "stepwise",
            "step_wise": "stepwise",
            "single_call": "single_call",
            "single": "single_call",
        }
        if "paradigm" not in df.columns:
            df["paradigm"] = df.apply(_derive_paradigm, axis=1)
        else:
            # normalize known aliases; blanks/unknown → NaN so they get backfilled
            df["paradigm"] = (
                df["paradigm"].astype(str).str.strip().str.lower().map(_PARADIGM_CANON)
            )
            mask = df["paradigm"].isna()
            if mask.any():
                df.loc[mask, "paradigm"] = df[mask].apply(_derive_paradigm, axis=1)
        self.df = df

        print(f"✅ Loaded {len(self.results_df)} runs ({len(self.df)} completed)")

    def _responses(self) -> list[tuple[str, bool]]:
        """Response variables present in the data, with their direction
        (True = higher is better)."""
        out: list[tuple[str, bool]] = []
        for r in QUALITY_RESPONSES:
            if r in self.df.columns:
                out.append((r, True))
        for r in COST_RESPONSES:
            if r in self.df.columns:
                out.append((r, False))
        return out

    def analyze(self) -> dict:
        """Run the full analysis."""
        responses = self._responses()
        # The reasoning paradigm is a 3-level factor only when single-call runs are
        # present; add it to the Friedman/Dunn comparison in that case.
        pairwise_factors = ["reasoner_model", "counter_model"]
        if "paradigm" in self.df.columns and self.df["paradigm"].nunique() > 2:
            pairwise_factors.append("paradigm")
        return {
            "summary": self._summary_stats(),
            "anova": {name: self._factorial_anova(name) for name, _ in responses},
            "pairwise_friedman": {
                factor: {
                    name: self._pairwise_comparison(factor, name, hb)
                    for name, hb in responses
                }
                for factor in pairwise_factors
            },
            "paradigm_summary": self._paradigm_summary(),
            "model_class_contrast": {
                name: self._wilcoxon_two_level("reasoner_class", "reasoning", name, hb)
                for name, hb in responses
            },
            "planning_wilcoxon": {
                name: self._wilcoxon_two_level("planning", True, name, hb)
                for name, hb in responses
            },
            "citation_faithfulness": self._citation_faithfulness(),
            "token_cost_analysis": self._token_cost_analysis(),
            "model_class_summary": self._model_class_summary(),
        }

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    def _summary_stats(self) -> dict:
        df = self.results_df
        completed = (
            int((df["status"] == "completed").sum())
            if "status" in df.columns
            else int(len(df))
        )
        failed = int((df["status"] == "failed").sum()) if "status" in df.columns else 0
        return {
            "total_runs": int(len(df)),
            "completed_runs": completed,
            "failed_runs": failed,
            "reasoner_models": sorted(
                self.df.get("reasoner_model", pd.Series()).unique().tolist()
            ),
            "counter_models": sorted(
                self.df.get("counter_model", pd.Series()).unique().tolist()
            ),
            "domains": sorted(self.df.get("domain", pd.Series()).unique().tolist()),
            "responses_analyzed": [name for name, _ in self._responses()],
        }

    # ------------------------------------------------------------------ #
    # Factorial ANOVA (parametric view; RQ2 interaction)
    # ------------------------------------------------------------------ #
    def _factorial_anova(self, response: str) -> dict:
        """Factorial ANOVA on `response`.

        Main effects of Reasoner, Counter, and planning, plus the
        Reasoner x Counter interaction (RQ2). Manual sum-of-squares
        decomposition with p-values from the F-distribution; the eta-squared
        importance index (SS/SST) is the effect size. Variability from the
        remaining factors (claim, replicate, other interactions) is pooled into
        the error term. See Appendix "Statistical Methods".
        """
        df = self.df
        needed = {"reasoner_model", "counter_model", response}
        if not needed.issubset(df.columns):
            return {"error": f"Missing columns; need {sorted(needed)}"}

        cols = ["reasoner_model", "counter_model", response]
        # Third factor: the 3-level reasoning paradigm when single-call runs are
        # present, otherwise the 2-level planning toggle. Using the boolean
        # `planning` with single-call runs would merge step-wise and single-call
        # (both planning=False) into a single level and hide the paradigm effect.
        third = None
        if "paradigm" in df.columns and df["paradigm"].nunique() > 2:
            third = "paradigm"
        elif "planning" in df.columns:
            third = "planning"
        if third:
            cols.append(third)
        d = df[cols].dropna()
        if d.empty:
            return {"error": "No data for ANOVA"}

        y = d[response].to_numpy(dtype=float)
        n_total = len(y)
        grand = float(y.mean())
        sst = float(((y - grand) ** 2).sum())
        if sst <= 0:
            return {"error": "Zero variance in response"}

        def _group_ss(by: Any) -> float:
            ss = 0.0
            for _, g in d.groupby(by):
                ss += len(g) * (float(g[response].mean()) - grand) ** 2
            return float(ss)

        a = int(d["reasoner_model"].nunique())
        b = int(d["counter_model"].nunique())
        ss_r = _group_ss("reasoner_model")
        ss_c = _group_ss("counter_model")
        ss_rc = _group_ss(["reasoner_model", "counter_model"]) - ss_r - ss_c

        effects: dict[str, tuple[float, int]] = {
            "reasoner": (ss_r, a - 1),
            "counter": (ss_c, b - 1),
            "reasoner_x_counter": (ss_rc, (a - 1) * (b - 1)),
        }
        model_df = (a - 1) + (b - 1) + (a - 1) * (b - 1)

        if third:
            p = int(d[third].nunique())
            if p > 1:
                effects[third] = (_group_ss(third), p - 1)
                model_df += p - 1

        ss_e = sst - sum(ss for ss, _ in effects.values())
        df_e = n_total - 1 - model_df
        if df_e <= 0:
            return {"error": "Insufficient replication for ANOVA error df"}
        mse = ss_e / df_e

        def _effect(ss: float, dof: int) -> dict:
            ms = ss / dof if dof > 0 else 0.0
            f = ms / mse if mse > 0 else float("inf")
            return {
                "ss": ss,
                "df": dof,
                "ms": ms,
                "F": float(f),
                "p_value": float(stats.f.sf(f, dof, df_e)),
                "eta_squared": float(ss / sst),
            }

        return {
            "response": response,
            "grand_mean": grand,
            "sst": sst,
            "error": {"ss": ss_e, "df": df_e, "ms": mse},
            "effects": {k: _effect(ss, dof) for k, (ss, dof) in effects.items()},
            "assumptions": self._anova_assumptions(d, response),
        }

    @staticmethod
    def _anova_assumptions(d: pd.DataFrame, response: str) -> dict:
        """Levene's homoscedasticity test across the Reasoner x Counter cells.
        Residual normality is assessed separately by the analyst (QQ plot). When
        the assumptions are doubtful, the distribution-free Friedman/Wilcoxon
        analysis (which does not depend on them) is the robust companion.
        """
        cells = [
            g[response].to_numpy(dtype=float)
            for _, g in d.groupby(["reasoner_model", "counter_model"])
            if len(g) > 1
        ]
        if len(cells) < 2:
            return {}
        lev_stat, lev_p = stats.levene(*cells)
        return {
            "homoscedasticity_levene": {
                "statistic": float(lev_stat),
                "p_value": float(lev_p),
                "homoscedastic": bool(lev_p > ALPHA),
            }
        }

    # ------------------------------------------------------------------ #
    # Friedman + Dunn/Holm pairwise comparison (RQ1, multi-level factors)
    # ------------------------------------------------------------------ #
    def _block_matrix(self, factor: str, response: str) -> dict[str, np.ndarray] | None:
        """Build the claim-blocked data for a factor: dict level -> array of
        per-claim mean values. The 22 claims are the paired blocks; the mean is
        taken over the marginalized factors and replicas. Only claims present for
        every level are kept (complete blocks, as Friedman requires)."""
        df = self.df
        if not {factor, "claim_id", response}.issubset(df.columns):
            return None
        piv = df.groupby(["claim_id", factor])[response].mean().unstack(factor)
        piv = piv.dropna(axis=0, how="any")
        if piv.shape[0] < 3 or piv.shape[1] < 2:
            return None
        return {str(level): piv[level].to_numpy(float) for level in piv.columns}

    @staticmethod
    def _holm(pvals: list[float]) -> np.ndarray:
        """Holm-Bonferroni step-down adjusted p-values."""
        p = np.asarray(pvals, float)
        m = len(p)
        order = np.argsort(p)
        adj = np.empty(m)
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, (m - rank) * p[idx])
            adj[idx] = min(running, 1.0)
        return adj

    def _dunn_holm(
        self, blocks: dict[str, np.ndarray]
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Dunn's test on Friedman rank sums with Holm correction. Returns the
        levels, the adjusted p-value matrix, and the mean-rank vector (lower mean
        rank = lower metric value)."""
        levels = list(blocks.keys())
        k = len(levels)
        M = np.column_stack([blocks[t] for t in levels])  # blocks x k
        R = M.shape[0]
        ranks = np.empty_like(M, float)
        for r in range(R):
            ranks[r] = pd.Series(M[r]).rank(method="average").to_numpy()
        mean_rank = ranks.sum(axis=0) / R
        se = np.sqrt(k * (k + 1) / (6.0 * R))
        pmat = np.ones((k, k))
        raw, pairs = [], []
        for i in range(k):
            for j in range(i + 1, k):
                z = abs(mean_rank[i] - mean_rank[j]) / se
                raw.append(2 * (1 - stats.norm.cdf(z)))
                pairs.append((i, j))
        adj = self._holm(raw)
        for (i, j), pa in zip(pairs, adj):
            pmat[i, j] = pmat[j, i] = float(pa)
        return levels, pmat, mean_rank

    def _pairwise_comparison(
        self, factor: str, response: str, higher_better: bool
    ) -> dict:
        """Friedman omnibus over the levels of `factor` (blocked by claim); when
        it rejects, Dunn post-hoc + Holm and a win/tie/loss matrix."""
        blocks = self._block_matrix(factor, response)
        if blocks is None:
            return {"error": "insufficient complete blocks"}
        levels = list(blocks.keys())
        arrs = [blocks[t] for t in levels]
        try:
            chi2, p_omni = stats.friedmanchisquare(*arrs)
        except ValueError as exc:
            return {"error": f"friedman failed: {exc}"}

        res: dict[str, Any] = {
            "factor": factor,
            "response": response,
            "n_blocks": int(len(arrs[0])),
            "levels": levels,
            "level_means": {t: float(np.mean(blocks[t])) for t in levels},
            "friedman": {
                "chi2": float(chi2),
                "p_value": float(p_omni),
                "rejected": bool(p_omni < ALPHA),
            },
        }
        if p_omni < ALPHA:
            levs, pmat, mean_rank = self._dunn_holm(blocks)
            k = len(levs)
            w = np.zeros((k, k), int)
            for i in range(k):
                for j in range(k):
                    if i == j or pmat[i, j] >= ALPHA:
                        continue
                    row_lower = mean_rank[i] < mean_rank[j]  # row has lower metric
                    row_better = (not row_lower) if higher_better else row_lower
                    w[i, j] = 1 if row_better else -1
            res["dunn_holm"] = {
                "mean_rank": {levs[i]: float(mean_rank[i]) for i in range(k)},
                "p_matrix": pmat.round(4).tolist(),
                "win_tie_loss": {"levels": levs, "matrix": w.tolist()},
                "tally": {
                    levs[i]: {
                        "wins": int((w[i] == 1).sum()),
                        "losses": int((w[i] == -1).sum()),
                        "ties": int(k - 1 - (w[i] != 0).sum()),
                    }
                    for i in range(k)
                },
            }
        return res

    # ------------------------------------------------------------------ #
    # Wilcoxon signed-rank for two-level contrasts (blocked by claim)
    # ------------------------------------------------------------------ #
    def _wilcoxon_two_level(
        self, factor: str, positive_level: Any, response: str, higher_better: bool
    ) -> dict:
        """Wilcoxon signed-rank test on the per-claim differences between the two
        levels of `factor` (e.g. reasoning vs instruction-tuned, planning on/off).
        `positive_level` is the level reported as the first term of the difference."""
        df = self.df
        if not {factor, "claim_id", response}.issubset(df.columns):
            return {"error": f"missing columns for {factor}"}
        if df[factor].nunique() != 2:
            return {"error": f"{factor} is not two-level"}
        piv = df.groupby(["claim_id", factor])[response].mean().unstack(factor)
        piv = piv.dropna(axis=0, how="any")
        if piv.shape[0] < 3:
            return {"error": "insufficient complete blocks"}
        levels = list(piv.columns)
        pos = positive_level if positive_level in levels else levels[0]
        neg = [lv for lv in levels if lv != pos][0]
        x = piv[pos].to_numpy(float)
        y = piv[neg].to_numpy(float)
        try:
            stat, p = stats.wilcoxon(x, y)
        except ValueError as exc:
            return {"error": f"wilcoxon failed: {exc}"}
        better = "tie"
        if p < ALPHA:
            pos_better = (float(x.mean()) > float(y.mean())) == higher_better
            better = str(pos) if pos_better else str(neg)
        return {
            "factor": factor,
            "response": response,
            "n_claims": int(len(x)),
            "levels": {"positive": str(pos), "negative": str(neg)},
            "means": {str(pos): float(x.mean()), str(neg): float(y.mean())},
            "statistic": float(stat),
            "p_value": float(p),
            "significant": bool(p < ALPHA),
            "better": better,
        }

    # ------------------------------------------------------------------ #
    # Citation faithfulness (RQ3): bootstrap CIs
    # ------------------------------------------------------------------ #
    def _citation_faithfulness(self) -> dict:
        df = self.df
        if df.empty or "citation_accuracy" not in df.columns:
            return {"error": "No citation_accuracy column / no data"}

        def _block(g: pd.DataFrame) -> dict:
            acc = g["citation_accuracy"].to_numpy(dtype=float)
            lo, hi = self._bootstrap_ci(acc)
            res = {
                "accuracy_mean": float(np.mean(acc)),
                "accuracy_std": float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0,
                "accuracy_ci_95": [float(lo), float(hi)],
                "n": int(len(acc)),
            }
            if {"citation_repaired", "citation_total"}.issubset(g.columns):
                res["repair_rate"] = float(
                    g["citation_repaired"].sum() / max(1, g["citation_total"].sum())
                )
            return res

        out: dict[str, Any] = {"overall": _block(df)}
        if "reasoner_class" in df.columns:
            out["by_class"] = {
                cls: _block(g) for cls, g in df.groupby("reasoner_class")
            }
        return out

    # ------------------------------------------------------------------ #
    # Token cost
    # ------------------------------------------------------------------ #
    def _token_cost_analysis(self) -> dict:
        df = self.df
        if df.empty or "total_tokens" not in df.columns:
            return {"error": "No token data"}

        group_col = (
            "reasoner_class" if "reasoner_class" in df.columns else "reasoner_model"
        )
        out: dict[str, Any] = {}
        for key, g in df.groupby(group_col):
            mean_tokens = float(g["total_tokens"].mean())
            block = {
                "tokens_per_run_mean": mean_tokens,
                "tokens_per_run_median": float(g["total_tokens"].median()),
                "tokens_per_quality_point": float(
                    mean_tokens / max(0.01, g["aqa_plausibility"].mean())
                ),
            }
            if "reasoning_tokens" in g.columns:
                block["reasoning_tokens_ratio"] = float(
                    g["reasoning_tokens"].mean() / max(1.0, mean_tokens)
                )
            out[str(key)] = block
        return out

    def _paradigm_summary(self) -> dict:
        """Descriptive means by reasoning paradigm (plan / stepwise / single-call).
        The formal 3-level contrast is the Friedman/Dunn test under
        `pairwise_friedman["paradigm"]` when single-call runs are present."""
        df = self.df
        if "paradigm" not in df.columns or df.empty:
            return {}
        cols = [
            c
            for c in (
                "aqa_plausibility",
                "aqa_cogency",
                "aqa_norm_support",
                "aqa_semantics",
                "citation_accuracy",
                "fidelity",
                "total_tokens",
            )
            if c in df.columns
        ]
        out: dict[str, Any] = {}
        for paradigm, g in df.groupby("paradigm"):
            row: dict[str, Any] = {"n": int(len(g))}
            for c in cols:
                row[f"{c}_mean"] = float(g[c].mean())
                row[f"{c}_std"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
            if "counter_abstained" in g.columns:
                row["abstention_rate"] = float(
                    g["counter_abstained"].astype(bool).mean()
                )
            out[str(paradigm)] = row
        return out

    def _model_class_summary(self) -> dict:
        """Descriptive means by model class (the formal contrast is the Wilcoxon
        test in `model_class_contrast`)."""
        df = self.df
        if df.empty or "reasoner_class" not in df.columns:
            return {}
        return {
            cls: {
                "aqa_plausibility_mean": float(g["aqa_plausibility"].mean()),
                "aqa_plausibility_std": (
                    float(g["aqa_plausibility"].std(ddof=1)) if len(g) > 1 else 0.0
                ),
                "reasoning_steps_mean": (
                    float(g["reasoning_steps"].mean())
                    if "reasoning_steps" in g.columns
                    else None
                ),
                "n": int(len(g)),
            }
            for cls, g in df.groupby("reasoner_class")
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _bootstrap_ci(
        values: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95
    ) -> tuple:
        """Percentile bootstrap confidence interval of the mean."""
        if len(values) == 0:
            return (float("nan"), float("nan"))
        resamples = np.random.choice(
            values, size=(n_bootstrap, len(values)), replace=True
        )
        means = np.mean(resamples, axis=1)
        lower = float(np.percentile(means, (1 - ci) / 2 * 100))
        upper = float(np.percentile(means, (1 + ci) / 2 * 100))
        return lower, upper

    def _plot_heatmaps(self, analysis: dict) -> None:
        """Render the win/tie/loss matrices as DeepSample-style heatmaps
        (white = row better, black = row worse, gray = n.s.). Skipped silently if
        matplotlib is unavailable."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.colors import BoundaryNorm, ListedColormap
        except Exception:
            return
        hdir = self.output_dir / "heatmaps"
        hdir.mkdir(exist_ok=True)
        cmap = ListedColormap(["#1a1a1a", "#bdbdbd", "#ffffff"])
        norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
        for factor, by_response in analysis.get("pairwise_friedman", {}).items():
            for response, res in by_response.items():
                wtl = res.get("dunn_holm", {}).get("win_tie_loss")
                if not wtl:
                    continue
                levels = wtl["levels"]
                w = np.asarray(wtl["matrix"], int)
                k = len(levels)
                fig, ax = plt.subplots(figsize=(0.7 * k + 1.5, 0.7 * k + 1.5))
                ax.imshow(w, cmap=cmap, norm=norm, aspect="equal")
                ax.set_xticks(range(k))
                ax.set_yticks(range(k))
                ax.set_xticklabels(levels, rotation=45, ha="right", fontsize=7)
                ax.set_yticklabels(levels, fontsize=7)
                for x in range(k + 1):
                    ax.axhline(x - 0.5, color="gray", lw=0.3)
                    ax.axvline(x - 0.5, color="gray", lw=0.3)
                ax.set_title(f"{factor} - {response}", fontsize=8)
                fig.tight_layout()
                fname = hdir / f"wtl_{factor}_{response}.pdf"
                fig.savefig(fname, bbox_inches="tight")
                plt.close(fig)
        print(f"📈 Heatmaps saved to {hdir}")

    def save_report(self, analysis: dict) -> None:
        """Save the analysis as JSON, render heatmaps, and print a short summary."""
        report_path = self.output_dir / "doe_analysis.json"
        with open(report_path, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\n📊 Analysis report saved to {report_path}")
        self._plot_heatmaps(analysis)

        print("\n" + "=" * 70)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 70)
        print(json.dumps(analysis.get("summary", {}), indent=2))

        anova = analysis.get("anova", {}).get("aqa_plausibility", {})
        if "effects" in anova:
            print("\n🎯 Factorial ANOVA (AQA plausibility):")
            for name, e in anova["effects"].items():
                print(
                    f"   {name:20s} F={e['F']:.2f} p={e['p_value']:.2e} "
                    f"eta^2={e['eta_squared']:.1%}"
                )

        fr = (
            analysis.get("pairwise_friedman", {})
            .get("reasoner_model", {})
            .get("aqa_plausibility", {})
        )
        if "friedman" in fr:
            print("\n🎯 RQ1 Reasoner models (Friedman, blocked by claim, AQA):")
            print(
                f"   chi2={fr['friedman']['chi2']:.2f} "
                f"p={fr['friedman']['p_value']:.2e} "
                f"rejected={fr['friedman']['rejected']}"
            )
            for lv, t in fr.get("dunn_holm", {}).get("tally", {}).items():
                print(f"   {lv:28s} W/T/L = {t['wins']}/{t['ties']}/{t['losses']}")

        mc = analysis.get("model_class_contrast", {}).get("aqa_plausibility", {})
        if "p_value" in mc:
            print(
                "\n🎯 RQ1 reasoning vs instruction-tuned (Wilcoxon signed-rank, AQA):"
            )
            print(f"   p={mc['p_value']:.2e}  better={mc['better']}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Analyze Multi-DoE results")
    parser.add_argument("--run-dir", required=True, help="DoE run directory")
    parser.add_argument("--output", required=True, help="Output directory for analysis")
    args = parser.parse_args()

    analyzer = AdvancedDoEAnalyzer(run_dir=args.run_dir, output_dir=args.output)
    analysis = analyzer.analyze()
    analyzer.save_report(analysis)
    print("✅ Analysis complete")


if __name__ == "__main__":
    main()
