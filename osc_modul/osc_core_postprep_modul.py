"""Conservative post-analysis outlier preprocess helpers for modal solvers."""

from __future__ import annotations

import numpy as np

POSTPREP_METHOD_HAMPEL = "hampel"
POSTPREP_METHOD_CHOICES = (POSTPREP_METHOD_HAMPEL,)

POSTPREP_REPAIR_MODE_MEDIAN = "median"
POSTPREP_REPAIR_MODE_LINEAR_INTERP = "linear_interp"
POSTPREP_REPAIR_MODE_CHOICES = (
    POSTPREP_REPAIR_MODE_MEDIAN,
    POSTPREP_REPAIR_MODE_LINEAR_INTERP,
)


def _longest_true_run(mask: np.ndarray) -> int:
    """Return longest consecutive-True run length."""

    m = np.asarray(mask, dtype=bool).reshape(-1)
    if m.size == 0:
        return 0
    best = 0
    cur = 0
    for v in m:
        if bool(v):
            cur += 1
            if cur > best:
                best = int(cur)
        else:
            cur = 0
    return int(best)


def mask_outliers_hampel(
    v_win: np.ndarray,
    *,
    hampel_half_window: int,
    hampel_nsigma: float,
) -> np.ndarray:
    """Build outlier mask using a short-window Hampel rule."""

    x = np.asarray(v_win, dtype=float).reshape(-1)
    n = int(x.size)
    if n == 0:
        return np.zeros(0, dtype=bool)

    half_w = int(max(1, hampel_half_window))
    n_sigma = float(max(0.0, hampel_nsigma))
    eps = 1e-12
    out = ~np.isfinite(x)

    for i in range(n):
        if not np.isfinite(x[i]):
            continue
        lo = int(max(0, i - half_w))
        hi = int(min(n, i + half_w + 1))
        w = x[lo:hi]
        wf = w[np.isfinite(w)]
        if wf.size < 3:
            continue
        med = float(np.median(wf))
        dev = float(abs(x[i] - med))
        mad = float(np.median(np.abs(wf - med)))
        scale = float(1.4826 * mad)

        if scale <= eps:
            # For near-constant local windows, only mark clear single-point excursions.
            close_cnt = int(np.count_nonzero(np.abs(wf - med) <= eps))
            if close_cnt >= int(max(3, np.floor(0.7 * wf.size))) and (dev > eps):
                out[i] = True
            continue

        if dev > (float(n_sigma) * scale):
            out[i] = True

    return np.asarray(out, dtype=bool).reshape(-1)


def _repair_outliers_local_median(
    v_win: np.ndarray,
    outlier_mask: np.ndarray,
    *,
    half_window: int,
) -> np.ndarray:
    """Repair masked points with local median from nearby non-masked points."""

    x = np.asarray(v_win, dtype=float).reshape(-1)
    mask = np.asarray(outlier_mask, dtype=bool).reshape(-1)
    n = int(x.size)
    out = np.array(x, copy=True)
    hw = int(max(1, half_window))
    for idx in np.flatnonzero(mask):
        lo = int(max(0, int(idx) - hw))
        hi = int(min(n, int(idx) + hw + 1))
        local = x[lo:hi]
        local_mask = mask[lo:hi]
        cand = local[(~local_mask) & np.isfinite(local)]
        if cand.size < 1:
            cand = local[np.isfinite(local)]
        if cand.size < 1:
            continue
        out[int(idx)] = float(np.median(cand))
    return out


def _repair_outliers_linear_interp(v_win: np.ndarray, outlier_mask: np.ndarray) -> np.ndarray:
    """Repair masked points with 1D linear interpolation on unmasked neighbors."""

    x = np.asarray(v_win, dtype=float).reshape(-1)
    mask = np.asarray(outlier_mask, dtype=bool).reshape(-1)
    n = int(x.size)
    out = np.array(x, copy=True)
    if n == 0:
        return out
    x_idx = np.arange(n, dtype=float)
    valid = (~mask) & np.isfinite(x)
    if np.count_nonzero(mask) <= 0:
        return out
    if np.count_nonzero(valid) >= 2:
        out[mask] = np.interp(x_idx[mask], x_idx[valid], x[valid])
        return out
    if np.count_nonzero(valid) == 1:
        out[mask] = float(x[valid][0])
    return out


def repair_sparse_outliers(
    v_win: np.ndarray,
    outlier_mask: np.ndarray,
    *,
    max_repair_points: int,
    max_repair_fraction: float,
    repair_mode: str,
    hampel_half_window: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Conditionally repair sparse outliers and return diagnostics."""

    x = np.asarray(v_win, dtype=float).reshape(-1)
    mask = np.asarray(outlier_mask, dtype=bool).reshape(-1)
    n = int(x.size)
    if mask.size != n:
        raise ValueError("outlier_mask size mismatch")

    detected = int(np.count_nonzero(mask))
    detected_frac = float(detected / max(1, n))
    max_pts = int(max(0, max_repair_points))
    max_frac = float(max(0.0, max_repair_fraction))
    longest_run = int(_longest_true_run(mask))
    run_limit = int(max(2, 2 * int(max(1, hampel_half_window))))

    edge_span = int(max(1, min(max(2, int(max(1, hampel_half_window))), max(1, n // 10))))
    start_hits = int(np.count_nonzero(mask[:edge_span]))
    end_hits = int(np.count_nonzero(mask[-edge_span:])) if n > 0 else 0
    edge_hits = int(start_hits + end_hits)

    suspicious_reasons: list[str] = []
    if detected > max_pts:
        suspicious_reasons.append("repair_points_exceeded")
    if detected_frac > max_frac:
        suspicious_reasons.append("repair_fraction_exceeded")
    if longest_run > run_limit:
        suspicious_reasons.append("long_outlier_run")
    if (edge_hits >= 2) and (edge_hits >= int(max(2, np.ceil(0.5 * max(1, detected))))):
        suspicious_reasons.append("edge_clustered_outliers")

    diag: dict[str, object] = {
        "detected_outlier_count": int(detected),
        "detected_outlier_fraction": float(detected_frac),
        "longest_outlier_run": int(longest_run),
        "edge_span_samples": int(edge_span),
        "edge_start_count": int(start_hits),
        "edge_end_count": int(end_hits),
        "suspicious": bool(len(suspicious_reasons) > 0),
        "suspicious_reason": str(suspicious_reasons[0]) if suspicious_reasons else "ok",
    }

    if detected <= 0:
        diag["status"] = "no_outliers"
        return np.array(x, copy=True), np.zeros_like(mask, dtype=bool), diag
    if suspicious_reasons:
        diag["status"] = "skipped_suspicious"
        return np.array(x, copy=True), np.zeros_like(mask, dtype=bool), diag

    mode = str(repair_mode).strip().lower()
    if mode == POSTPREP_REPAIR_MODE_LINEAR_INTERP:
        repaired = _repair_outliers_linear_interp(x, mask)
    else:
        repaired = _repair_outliers_local_median(
            x,
            mask,
            half_window=int(max(1, hampel_half_window)),
        )
    finite_pair = np.isfinite(x) & np.isfinite(repaired)
    edited_mask = np.asarray(mask & finite_pair & (np.abs(repaired - x) > 0.0), dtype=bool)

    if int(np.count_nonzero(edited_mask)) <= 0:
        diag["status"] = "repair_no_effect"
        return np.array(x, copy=True), np.zeros_like(mask, dtype=bool), diag

    diag["status"] = "applied"
    return repaired, edited_mask, diag


def summarize_postprep(
    *,
    n_samples: int,
    edited_mask: np.ndarray,
    detected_mask: np.ndarray,
    method: str,
    repair_mode: str,
    status: str,
    suspicious: bool,
    suspicious_reason: str,
    keep_raw_summary: bool,
    diagnostics: dict[str, object],
) -> dict[str, object]:
    """Build compact record fields for post-analysis preprocess."""

    n = int(max(0, n_samples))
    edited = int(np.count_nonzero(np.asarray(edited_mask, dtype=bool)))
    detected = int(np.count_nonzero(np.asarray(detected_mask, dtype=bool)))
    out = {
        "postprep_applied": bool(str(status) == "applied"),
        "postprep_method": str(method),
        "postprep_repair_mode": str(repair_mode),
        "postprep_status": str(status),
        "postprep_outlier_count": int(edited),
        "postprep_outlier_fraction": float(edited / max(1, n)),
        "postprep_suspicious": bool(suspicious),
        "postprep_suspicious_reason": str(suspicious_reason),
    }
    if bool(keep_raw_summary):
        out.update(
            {
                "postprep_window_samples": int(n),
                "postprep_detected_outlier_count": int(detected),
                "postprep_detected_outlier_fraction": float(detected / max(1, n)),
                "postprep_longest_outlier_run": int(diagnostics.get("longest_outlier_run", 0)),
                "postprep_edge_span_samples": int(diagnostics.get("edge_span_samples", 0)),
                "postprep_edge_start_count": int(diagnostics.get("edge_start_count", 0)),
                "postprep_edge_end_count": int(diagnostics.get("edge_end_count", 0)),
            }
        )
    return out


def default_postprep_summary(
    *,
    enabled: bool,
    method: str,
    repair_mode: str,
    keep_raw_summary: bool,
) -> dict[str, object]:
    """Return default no-op postprep fields for skipped records."""

    out = {
        "postprep_applied": False,
        "postprep_method": str(method),
        "postprep_repair_mode": str(repair_mode),
        "postprep_status": ("disabled" if (not bool(enabled)) else "not_run"),
        "postprep_outlier_count": 0,
        "postprep_outlier_fraction": 0.0,
        "postprep_suspicious": False,
        "postprep_suspicious_reason": "none",
    }
    if bool(keep_raw_summary):
        out.update(
            {
                "postprep_window_samples": 0,
                "postprep_detected_outlier_count": 0,
                "postprep_detected_outlier_fraction": 0.0,
                "postprep_longest_outlier_run": 0,
                "postprep_edge_span_samples": 0,
                "postprep_edge_start_count": 0,
                "postprep_edge_end_count": 0,
            }
        )
    return out


def apply_modal_postprep(
    v_win: np.ndarray,
    *,
    enabled: bool,
    method: str,
    hampel_half_window: int,
    hampel_nsigma: float,
    max_repair_points: int,
    max_repair_fraction: float,
    repair_mode: str,
    keep_raw_summary: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Apply conservative outlier postprep for one selected modal-analysis window."""

    x = np.asarray(v_win, dtype=float).reshape(-1)
    method_eff = str(method).strip().lower()
    repair_mode_eff = str(repair_mode).strip().lower()
    if not bool(enabled):
        summary = default_postprep_summary(
            enabled=False,
            method=method_eff,
            repair_mode=repair_mode_eff,
            keep_raw_summary=bool(keep_raw_summary),
        )
        return np.array(x, copy=True), np.zeros_like(x, dtype=bool), summary

    if method_eff != POSTPREP_METHOD_HAMPEL:
        summary = default_postprep_summary(
            enabled=True,
            method=method_eff,
            repair_mode=repair_mode_eff,
            keep_raw_summary=bool(keep_raw_summary),
        )
        summary["postprep_status"] = "unsupported_method"
        summary["postprep_suspicious"] = True
        summary["postprep_suspicious_reason"] = "unsupported_method"
        return np.array(x, copy=True), np.zeros_like(x, dtype=bool), summary

    detected_mask = mask_outliers_hampel(
        x,
        hampel_half_window=int(max(1, hampel_half_window)),
        hampel_nsigma=float(max(0.0, hampel_nsigma)),
    )
    x_repaired, edited_mask, diag = repair_sparse_outliers(
        x,
        detected_mask,
        max_repair_points=int(max(0, max_repair_points)),
        max_repair_fraction=float(max(0.0, max_repair_fraction)),
        repair_mode=repair_mode_eff,
        hampel_half_window=int(max(1, hampel_half_window)),
    )
    status = str(diag.get("status", "unknown"))
    suspicious = bool(diag.get("suspicious", False))
    suspicious_reason = str(diag.get("suspicious_reason", "unknown"))
    summary = summarize_postprep(
        n_samples=int(x.size),
        edited_mask=edited_mask,
        detected_mask=detected_mask,
        method=method_eff,
        repair_mode=repair_mode_eff,
        status=status,
        suspicious=suspicious,
        suspicious_reason=suspicious_reason,
        keep_raw_summary=bool(keep_raw_summary),
        diagnostics=diag,
    )
    return x_repaired, edited_mask, summary


__all__ = [
    "POSTPREP_METHOD_HAMPEL",
    "POSTPREP_METHOD_CHOICES",
    "POSTPREP_REPAIR_MODE_MEDIAN",
    "POSTPREP_REPAIR_MODE_LINEAR_INTERP",
    "POSTPREP_REPAIR_MODE_CHOICES",
    "mask_outliers_hampel",
    "repair_sparse_outliers",
    "summarize_postprep",
    "default_postprep_summary",
    "apply_modal_postprep",
]
