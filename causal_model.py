"""
Causal inference engine using DoWhy + EconML.

Provides:
  - Average Treatment Effect (ATE) estimation via DoWhy backdoor
  - Heterogeneous Treatment Effect (HTE) via EconML LinearDML
  - Refutation tests for trustworthiness indicators
"""

from __future__ import annotations

import logging
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import dowhy
from dowhy import CausalModel

from causal_graph import build_dag, get_gml_string, TREATMENTS, OUTCOMES

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Refutation pass/fail thresholds (fraction of original estimate)
PLACEBO_THRESHOLD = 0.5       # placebo effect must be < 50% of original
RANDOM_CAUSE_THRESHOLD = 0.15 # effect shift must be < 15% of original
DATA_SUBSET_THRESHOLD = 0.2   # effect shift must be < 20% of original

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent


def _load_data() -> pd.DataFrame:
    return pd.read_csv(_ROOT / "data" / "lifestyle_data.csv")


@lru_cache(maxsize=1)
def _cached_gml() -> str:
    """Build DAG and return GML string, cached across calls."""
    return get_gml_string(build_dag())


def _common_causes(treatment: str) -> dict[str, list[str]]:
    """Return confounders (W) and effect modifiers (X) for a given treatment.

    W = variables that confound the treatment-outcome relationship (controls).
    X = variables that modify the treatment effect (heterogeneity sources).
    These are used by EconML's LinearDML; DoWhy uses the graph structure instead.
    """
    config = {
        "sleep":        {"W": ["screen_time", "caffeine", "stress", "work_hours"], "X": ["age"]},
        "exercise":     {"W": ["work_hours"],                                      "X": ["age"]},
        "diet_quality": {"W": ["exercise", "work_hours"],                          "X": ["age"]},
        "screen_time":  {"W": ["work_hours"],                                      "X": ["age"]},
        "caffeine":     {"W": ["work_hours"],                                      "X": ["age"]},
    }
    return config.get(treatment, {"W": ["work_hours"], "X": ["age"]})


# ---------------------------------------------------------------------------
# ATE estimation
# ---------------------------------------------------------------------------

def estimate_ate(
    treatment: str,
    outcome: str,
    data: pd.DataFrame | None = None,
    full: bool = True,
) -> dict:
    """
    Estimate the Average Treatment Effect of `treatment` on `outcome`
    using DoWhy's backdoor linear regression estimator.

    Parameters
    ----------
    full : bool
        If True, compute confidence intervals and p-values (slower).
        If False, return only the point estimate (fast path for tables).

    Returns dict with keys: estimate, p_value, ci_low, ci_high, method.
    """
    if data is None:
        data = _load_data()

    gml = _cached_gml()

    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        graph=gml,
    )

    identified = model.identify_effect(proceed_when_unidentifiable=True)

    estimate = model.estimate_effect(
        identified,
        method_name="backdoor.linear_regression",
        confidence_intervals=full,
        test_significance=full,
    )

    ate_value = estimate.value
    ci_low, ci_high, p_val = None, None, None

    if full:
        ci = estimate.get_confidence_intervals()
        p_val = estimate.test_stat_significance()

        if isinstance(ci, np.ndarray):
            if ci.ndim == 2:
                ci_low, ci_high = float(ci[0][0]), float(ci[0][1])
            else:
                ci_low, ci_high = float(ci[0]), float(ci[1])
        elif isinstance(ci, (list, tuple)):
            ci_low, ci_high = float(ci[0]), float(ci[1])

        if isinstance(p_val, dict):
            p_val = list(p_val.values())[0]
        if isinstance(p_val, (np.ndarray,)):
            p_val = float(p_val.flat[0])

    return {
        "estimate": float(ate_value),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": float(p_val) if p_val is not None else None,
        "method": "Backdoor (Linear Regression)",
    }


# ---------------------------------------------------------------------------
# Refutation tests
# ---------------------------------------------------------------------------

def run_refutations(
    treatment: str,
    outcome: str,
    data: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Run DoWhy refutation tests and return a list of result dicts.
    Each dict: {test_name, estimated_effect, new_effect, p_value, passed}
    """
    if data is None:
        data = _load_data()

    gml = _cached_gml()

    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        graph=gml,
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified,
        method_name="backdoor.linear_regression",
    )

    results = []

    # 1. Placebo treatment
    try:
        ref_placebo = model.refute_estimate(
            identified, estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute",
            num_simulations=50,
        )
        placebo_effect = ref_placebo.new_effect
        if isinstance(placebo_effect, (np.ndarray,)):
            placebo_effect = float(placebo_effect.flat[0])
        p_val = getattr(ref_placebo, "refutation_result", {})
        if isinstance(p_val, dict):
            p_val = p_val.get("p_value", None)
        else:
            p_val = None
        results.append({
            "test_name": "Placebo Treatment",
            "description": "Replace treatment with random noise — effect should vanish",
            "estimated_effect": float(estimate.value),
            "new_effect": float(placebo_effect),
            "passed": abs(float(placebo_effect)) < abs(float(estimate.value)) * PLACEBO_THRESHOLD,
        })
    except Exception:
        results.append({
            "test_name": "Placebo Treatment",
            "description": "Replace treatment with random noise — effect should vanish",
            "estimated_effect": float(estimate.value),
            "new_effect": None,
            "passed": None,
        })

    # 2. Random common cause
    try:
        ref_rcc = model.refute_estimate(
            identified, estimate,
            method_name="random_common_cause",
        )
        rcc_effect = ref_rcc.new_effect
        if isinstance(rcc_effect, (np.ndarray,)):
            rcc_effect = float(rcc_effect.flat[0])
        results.append({
            "test_name": "Random Common Cause",
            "description": "Add a random confounder — effect should remain stable",
            "estimated_effect": float(estimate.value),
            "new_effect": float(rcc_effect),
            "passed": abs(float(rcc_effect) - float(estimate.value)) < abs(float(estimate.value)) * RANDOM_CAUSE_THRESHOLD,
        })
    except Exception:
        results.append({
            "test_name": "Random Common Cause",
            "description": "Add a random confounder — effect should remain stable",
            "estimated_effect": float(estimate.value),
            "new_effect": None,
            "passed": None,
        })

    # 3. Data subset refuter
    try:
        ref_sub = model.refute_estimate(
            identified, estimate,
            method_name="data_subset_refuter",
            subset_fraction=0.8,
            num_simulations=50,
        )
        sub_effect = ref_sub.new_effect
        if isinstance(sub_effect, (np.ndarray,)):
            sub_effect = float(sub_effect.flat[0])
        results.append({
            "test_name": "Data Subset",
            "description": "Use 80% of data — effect should remain stable",
            "estimated_effect": float(estimate.value),
            "new_effect": float(sub_effect),
            "passed": abs(float(sub_effect) - float(estimate.value)) < abs(float(estimate.value)) * DATA_SUBSET_THRESHOLD,
        })
    except Exception:
        results.append({
            "test_name": "Data Subset",
            "description": "Use 80% of data — effect should remain stable",
            "estimated_effect": float(estimate.value),
            "new_effect": None,
            "passed": None,
        })

    return results


# ---------------------------------------------------------------------------
# What-If estimation (personalized via EconML LinearDML)
# ---------------------------------------------------------------------------

def estimate_whatif(
    treatment: str,
    outcome: str,
    user_profile: dict[str, float],
    delta: float = 1.0,
    data: pd.DataFrame | None = None,
) -> dict:
    """
    Estimate the personalised what-if effect for a user:
    'What if I change `treatment` by `delta` units?'

    Uses EconML LinearDML for heterogeneous treatment effects.

    Returns dict: {treatment, outcome, delta, predicted_change, user_profile}
    """
    if data is None:
        data = _load_data()

    from econml.dml import LinearDML
    from sklearn.linear_model import RidgeCV

    # Separate confounders (W) from effect modifiers (X)
    config = _common_causes(treatment)
    effect_modifiers = [c for c in config["X"] if c in data.columns]
    confounders = [c for c in config["W"] if c in data.columns]

    X = data[effect_modifiers].values if effect_modifiers else None
    W = data[confounders].values if confounders else None
    T = data[[treatment]].values.ravel()
    Y = data[[outcome]].values.ravel()

    try:
        dml = LinearDML(
            model_y=RidgeCV(),
            model_t=RidgeCV(),
            random_state=42,
        )
        dml.fit(Y, T, X=X, W=W)

        # Build user feature vector for effect modifiers
        if effect_modifiers:
            user_x = np.array([[user_profile.get(c, data[c].mean()) for c in effect_modifiers]])
        else:
            user_x = None

        # Estimate marginal effect per 1-unit change, then scale by delta
        cate_per_unit = dml.effect(user_x, T0=0, T1=1)
        cate_value = float(cate_per_unit.flat[0]) * delta

        # Confidence intervals — swap bounds when delta < 0 to keep low < high
        try:
            ci = dml.effect_interval(user_x, T0=0, T1=1, alpha=0.05)
            bound_a = float(ci[0].flat[0]) * delta
            bound_b = float(ci[1].flat[0]) * delta
            ci_low, ci_high = min(bound_a, bound_b), max(bound_a, bound_b)
        except Exception:
            ci_low, ci_high = None, None

    except Exception as e:
        logger.warning("LinearDML failed for %s → %s: %s. Falling back to ATE.", treatment, outcome, e)
        ate = estimate_ate(treatment, outcome, data)
        cate_value = ate["estimate"] * delta
        if ate["ci_low"] is not None and ate["ci_high"] is not None:
            bound_a = ate["ci_low"] * delta
            bound_b = ate["ci_high"] * delta
            ci_low, ci_high = min(bound_a, bound_b), max(bound_a, bound_b)
        else:
            ci_low, ci_high = None, None

    return {
        "treatment": treatment,
        "outcome": outcome,
        "delta": delta,
        "predicted_change": round(cate_value, 3),
        "ci_low": round(ci_low, 3) if ci_low is not None else None,
        "ci_high": round(ci_high, 3) if ci_high is not None else None,
    }


# ---------------------------------------------------------------------------
# Batch what-if across all outcomes
# ---------------------------------------------------------------------------

def estimate_whatif_all_outcomes(
    treatment: str,
    user_profile: dict[str, float],
    delta: float = 1.0,
    data: pd.DataFrame | None = None,
) -> list[dict]:
    """Run what-if for all outcomes and return list of result dicts."""
    if data is None:
        data = _load_data()
    results = []
    for outcome in OUTCOMES:
        r = estimate_whatif(treatment, outcome, user_profile, delta, data)
        results.append(r)
    return results
