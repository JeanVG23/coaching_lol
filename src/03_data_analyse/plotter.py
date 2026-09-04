import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess

# ==========================================
# 0. DIAGNOSTIC SHAP PAR FEATURE (LOWESS)
# ==========================================
_LOWESS_MAX_SAMPLES = 5000
_LOWESS_FRAC = 0.25
_SIGNIFICANCE_PCT = 0.05


class _Diag:
    """Contexte de diagnostic d'une feature + fabrique de verdict.

    Les 12 `return {...}` du module portaient la même forme (trend_type,
    diagnostic, effect_strength, confidence = k × qualité, puis les 3 blocs de
    métriques fusionnés) : ajouter une métrique de risque demandait d'éditer 12
    littéraux, et un oubli produisait un diagnostic à clé manquante consommé sans
    erreur par shap_analysis. Les 4 détecteurs prenaient en plus les mêmes 4
    valeurs en paramètres positionnels.
    """

    __slots__ = ("robustness", "risk", "interaction", "quality")

    def __init__(self, robustness: dict, risk: dict, interaction: dict, quality: float):
        self.robustness = robustness
        self.risk = risk
        self.interaction = interaction
        self.quality = quality

    def verdict(self, trend_type: str, diagnostic: str, effect_strength: float,
                conf_mult: float, **extra) -> dict:
        return {
            "trend_type": trend_type,
            "diagnostic": diagnostic,
            "effect_strength": effect_strength,
            **extra,
            "confidence": conf_mult * self.quality,
            **self.robustness,
            **self.risk,
            **self.interaction,
        }


def _compute_shap_risk_metrics(s: np.ndarray) -> dict:
    _mean_abs_shap = float(np.mean(np.abs(s)))
    _tail_risk_95 = float(np.quantile(np.abs(s), 0.95))
    _downside_risk = float(np.quantile(s, 0.05))
    _s_std = float(np.std(s))

    if _downside_risk < -0.05 * _s_std:
        _severe_neg_mask = s <= np.quantile(s, 0.05)
        _severe_negative_mean = float(s[_severe_neg_mask].mean())
    else:
        _severe_negative_mean = 0.0

    return {
        "mean_abs_shap": _mean_abs_shap,
        "tail_risk_95": _tail_risk_95,
        "downside_risk": _downside_risk,
        "severe_negative_mean": _severe_negative_mean,
    }


# (trend_type, diagnostic) d'une feature binaire, selon le signe de l'écart.
_BINARY_VERDICTS = {
    0: ("neutral", "⚪ NEUTRE"),
    1: ("binary_positive", "🟢 BONUS (Recommandé)"),
    -1: ("binary_negative", "🔴 MALUS (Pénalisant)"),
}


def _handle_binary_feature(x: np.ndarray, s: np.ndarray, unique_vals: np.ndarray, _risk: dict) -> dict:
    _robustness = {"stability": 1.0, "coverage": min(1.0, len(x) / 150), "density": 1.0}
    _quality = 0.4 * _robustness["stability"] + 0.3 * _robustness["coverage"] + 0.3 * _robustness["density"]
    diag = _Diag(_robustness, _risk,
                 {"interaction_ratio": 0.0, "has_strong_interaction": False}, _quality)

    low_val, high_val = np.min(unique_vals), np.max(unique_vals)
    mean_low = s[x == low_val].mean() if np.any(x == low_val) else 0
    mean_high = s[x == high_val].mean() if np.any(x == high_val) else 0
    diff = mean_high - mean_low

    if abs(diff) < _SIGNIFICANCE_PCT * np.std(s):
        sign = 0
    else:
        sign = 1 if diff > 0 else -1
    trend, label = _BINARY_VERDICTS[sign]
    return diag.verdict(trend, label, float(abs(diff)), 0.95)


def _prepare_data_for_lowess(x: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    x_low, x_high = np.percentile(x, [1, 99])
    keep = (x >= x_low) & (x <= x_high)
    x_win, s_win = x[keep], s[keep]

    if len(x_win) < 20:
        return x_win, s_win, False

    if len(x_win) > _LOWESS_MAX_SAMPLES:
        idx = np.random.default_rng(42).choice(len(x_win), _LOWESS_MAX_SAMPLES, replace=False)
        idx.sort()
        x_win, s_win = x_win[idx], s_win[idx]

    order = np.argsort(x_win)
    return x_win[order], s_win[order], True


def _compute_lowess_smoothing(x: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_unique_ratio = len(np.unique(x)) / len(x)
    frac = max(0.15, min(_LOWESS_FRAC, n_unique_ratio * 0.5))

    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = lowess(s, x, frac=frac, return_sorted=True)
    return smoothed[:, 0], smoothed[:, 1]


def _compute_robustness_metrics(x: np.ndarray, s: np.ndarray, s_smooth: np.ndarray) -> tuple[dict, float, np.ndarray]:
    _residuals = s - s_smooth
    _stability = max(0.0, 1.0 - float(np.var(_residuals) / (np.var(s) + 1e-8)))
    _coverage = min(1.0, len(x) / 150)
    _max_gap = float(np.max(np.diff(x)) / (x[-1] - x[0] + 1e-8))
    _density = max(0.0, 1.0 - _max_gap * 5)
    _robustness = {"stability": _stability, "coverage": _coverage, "density": _density}
    _quality = 0.4 * _stability + 0.3 * _coverage + 0.3 * _density
    return _robustness, _quality, _residuals


def _compute_interaction_metrics(x: np.ndarray, s: np.ndarray, _residuals: np.ndarray, amp: float) -> dict:
    _n_bins = min(10, max(3, len(x) // 30))
    _interaction_bins = np.array_split(np.arange(len(x)), _n_bins)
    _min_bin_size = max(15, int(0.05 * len(x)))

    _valid_local_stds = []
    for _bin_idx in _interaction_bins:
        if len(_bin_idx) >= _min_bin_size:
            _valid_local_stds.append(float(np.std(_residuals[_bin_idx])))

    if _valid_local_stds:
        _mean_local_std = float(np.mean(_valid_local_stds))
        _interaction_ratio = float(_mean_local_std / (amp + 1e-8))
        _global_std = float(np.std(s))
        _is_absolutely_high = _mean_local_std > (0.50 * _global_std)
        _is_relatively_high = _interaction_ratio > 0.75
        _has_strong_interaction = bool(_is_absolutely_high and _is_relatively_high and amp > 0.05)
    else:
        _interaction_ratio = 0.0
        _has_strong_interaction = False

    return {
        "interaction_ratio": _interaction_ratio,
        "has_strong_interaction": _has_strong_interaction,
    }


def _evaluate_directional_fallback(s_smooth: np.ndarray, amp: float,
                                   significance: float, diag: "_Diag") -> dict:
    n = len(s_smooth)
    third = max(1, n // 3)
    diff = np.mean(s_smooth[-third:]) - np.mean(s_smooth[:third])

    if diff > significance:
        return diag.verdict("weak_positive", "📈 TENDANCE POSITIVE (Avoir plus aide)",
                            amp, 0.50)
    if diff < -significance:
        return diag.verdict("weak_negative", "📉 TENDANCE NÉGATIVE (Avoir plus pénalise)",
                            amp, 0.50)
    return diag.verdict("neutral", "⚪ NEUTRE", amp, 0.30)


def _detect_peak_zone(x_smooth: np.ndarray, s_smooth: np.ndarray, amp: float,
                      peak_idx: int, n: int, diag: "_Diag") -> dict | None:
    if 0.15 < peak_idx / n < 0.85:
        plateau_threshold = 0.10
        top_mask = s_smooth >= (np.max(s_smooth) - plateau_threshold * amp)
        if np.any(top_mask):
            opt_min = float(np.min(x_smooth[top_mask]))
            opt_max = float(np.max(x_smooth[top_mask]))
            full_range = float(x_smooth[-1] - x_smooth[0])
            zone_width = opt_max - opt_min
            if full_range > 0 and zone_width / full_range < 0.50:
                fmt = ".2f" if abs(opt_max) < 10 else ".0f"
                return diag.verdict(
                    "optimal_range",
                    f"🎯 ZONE IDÉALE (Cible : ~{opt_min:{fmt}} à {opt_max:{fmt}})",
                    amp, 0.80,
                    peak_x=float(x_smooth[peak_idx]),
                    optimal_range=(opt_min, opt_max))
    return None


def _detect_valley_zone(s: np.ndarray, x_smooth: np.ndarray, s_smooth: np.ndarray,
                        amp: float, valley_idx: int, n: int,
                        diag: "_Diag") -> dict | None:
    if 0.15 < valley_idx / n < 0.85:
        valley_x = float(x_smooth[valley_idx])
        valley_shap = float(s_smooth[valley_idx])
        vfmt = ".2f" if abs(valley_x) < 10 else ".0f"

        peak_left = float(np.max(s_smooth[:valley_idx])) if valley_idx > 0 else valley_shap
        peak_right = float(np.max(s_smooth[valley_idx + 1 :])) if valley_idx < n - 1 else valley_shap
        depth = max(peak_left, peak_right) - valley_shap

        if depth >= 0.30 * amp:
            min_support = max(5, int(0.15 * n))
            if valley_idx < min_support or (n - valley_idx - 1) < min_support:
                return diag.verdict(
                    "u_shape", f"🌀 EFFET EN U (Creux vers ~{valley_x:{vfmt}})", amp, 0.55,
                    valley_x=valley_x, dominant_tail="ambiguous",
                    sweet_spot_x=None, tail_shap_diff=0.0)

            n_quarter = max(5, n // 4)
            left_stable = float(np.median(s_smooth[:n_quarter]))
            right_stable = float(np.median(s_smooth[-n_quarter:]))
            tail_diff = right_stable - left_stable
            threshold = max(0.25 * amp, 0.05 * np.std(s))

            if tail_diff > threshold:
                dominant_tail = "high"
                peak_x = float(x_smooth[valley_idx + 1 + np.argmax(s_smooth[valley_idx + 1 :])])
                pfmt = ".2f" if abs(peak_x) < 10 else ".0f"
                diagnostic = f"🌀 EFFET EN U — Préférer la zone haute (Cible optimale vers ~{peak_x:{pfmt}})"
                _base_conf = 0.70
            elif tail_diff < -threshold:
                dominant_tail = "low"
                peak_x = float(x_smooth[np.argmax(s_smooth[:valley_idx])])
                pfmt = ".2f" if abs(peak_x) < 10 else ".0f"
                diagnostic = f"🌀 EFFET EN U — Préférer la zone basse (Cible optimale vers ~{peak_x:{pfmt}})"
                _base_conf = 0.70
            else:
                dominant_tail = "ambiguous"
                peak_x = None
                diagnostic = f"🌀 EFFET EN U (Creux vers ~{valley_x:{vfmt}})"
                _base_conf = 0.55

            return diag.verdict(
                "u_shape", diagnostic, amp, _base_conf,
                valley_x=valley_x, dominant_tail=dominant_tail,
                sweet_spot_x=peak_x, tail_shap_diff=float(tail_diff))
    return None


def _detect_shape_and_diagnostic(x: np.ndarray, s: np.ndarray, x_smooth: np.ndarray,
                                 s_smooth: np.ndarray, amp: float, significance: float,
                                 diag: "_Diag") -> dict:
    if amp < significance:
        return diag.verdict("neutral", "⚪ NEUTRE", amp, 0.30)

    ds = np.diff(s_smooth)
    eps = significance / 10
    pos_ratio = np.mean(ds > eps)
    neg_ratio = np.mean(ds < -eps)
    with np.errstate(invalid="ignore"):
        rho_raw, _ = spearmanr(x, s)
        rho = float(rho_raw)
    if not np.isfinite(rho):        # spearmanr renvoie nan sur une série constante
        rho = 0.0

    if pos_ratio > 0.75 and rho > 0.3:
        return diag.verdict("monotonic_increasing", "📈 À MAXIMISER (Le plus possible)",
                            amp, float(min(1.0, abs(rho))), spearman_rho=float(rho))

    if neg_ratio > 0.75 and rho < -0.3:
        return diag.verdict("monotonic_decreasing", "📉 À MINIMISER (Toxique / Sur-opti)",
                            amp, float(min(1.0, abs(rho))), spearman_rho=float(rho))

    peak_idx = int(np.argmax(s_smooth))
    valley_idx = int(np.argmin(s_smooth))
    n = len(s_smooth)

    peak_res = _detect_peak_zone(x_smooth, s_smooth, amp, peak_idx, n, diag)
    if peak_res:
        return peak_res

    valley_res = _detect_valley_zone(s, x_smooth, s_smooth, amp, valley_idx, n, diag)
    if valley_res:
        return valley_res

    return _evaluate_directional_fallback(s_smooth, amp, significance, diag)


def compute_shap_trend_summary(x_vals: np.ndarray, s_vals: np.ndarray) -> dict:
    x_vals = np.asarray(x_vals, dtype=float)
    s_vals = np.asarray(s_vals, dtype=float)

    mask = np.isfinite(x_vals) & np.isfinite(s_vals)
    x = x_vals[mask]
    s = s_vals[mask]

    if len(x) < 20:
        return {
            "trend_type": "insufficient_data",
            "diagnostic": "⚪ DONNÉES INSUFFISANTES",
            "confidence": 0.0,
        }

    if not np.issubdtype(x.dtype, np.number) or np.std(x) == 0:
        return {"trend_type": "neutral", "diagnostic": "⚪ NEUTRE", "confidence": 0.0}

    _risk = _compute_shap_risk_metrics(s)

    unique_vals = np.unique(x)
    if len(unique_vals) <= 2:
        return _handle_binary_feature(x, s, unique_vals, _risk)

    x, s, is_valid = _prepare_data_for_lowess(x, s)
    if not is_valid:
        return {
            "trend_type": "insufficient_data",
            "diagnostic": "⚪ DONNÉES INSUFFISANTES",
            "confidence": 0.0,
            "stability": 0.0,
            "coverage": 0.0,
            "density": 0.0,
            **_risk,
        }

    x_smooth, s_smooth = _compute_lowess_smoothing(x, s)

    _robustness, _quality, _residuals = _compute_robustness_metrics(x, s, s_smooth)
    amp = float(np.max(s_smooth) - np.min(s_smooth))
    significance = max(_SIGNIFICANCE_PCT * np.std(s), 1e-8)
    _interaction = _compute_interaction_metrics(x, s, _residuals, amp)

    diag = _Diag(_robustness, _risk, _interaction, _quality)
    return _detect_shape_and_diagnostic(x, s, x_smooth, s_smooth, amp, significance, diag)


# ==========================================
# 1. GÉNÉRATION DES DIAGNOSTICS LOL
# ==========================================
def generate_lol_diagnostics(X: pd.DataFrame, shap_values: np.ndarray, features: list[str]) -> list[dict]:
    """
    Applique le moteur de diagnostic LOWESS sur chaque feature du dataset LoL.
    Retourne une liste de dictionnaires contenant le nom de la feature et son diagnostic détaillé.
    """
    results = []
    
    # On itère sur chaque feature
    for i, col in enumerate(features):
        x_vals = X[col].values
        s_vals = shap_values[:, i]
        
        # Appel de la fonction de diagnostic LOWESS
        diag = compute_shap_trend_summary(x_vals, s_vals)
        
        results.append({
            "feature": col,
            "trend_type": diag.get("trend_type", "unknown"),
            "diagnostic": diag.get("diagnostic", "⚪ Inconnu"),
            "confidence": diag.get("confidence", 0.0),
            "effect_strength": diag.get("effect_strength", 0.0)
        })
    # Tri par importance (effet strength ou confiance)
    results = sorted(results, key=lambda x: x["effect_strength"], reverse=True)
    return results
