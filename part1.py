"""Compute the Part 1 statistical and regression results for the capstone."""

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "records.csv"
OUTPUT_PATH = ROOT / "outputs" / "part1_results.txt"

REQUIRED_COLUMNS = (
    "months_registered",
    "missed_visit_count_12m",
    "copay_burden_ratio",
    "avg_bill_inr_thousands",
    "adherence_score",
    "chronic_condition_flag",
    "insured_flag",
    "care_plan_cost_inr_thousands",
)


def load_records(path: Path) -> dict[str, np.ndarray]:
    """Load and validate the numeric columns needed for Part 1."""
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a CSV header.")

        missing_columns = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing_columns)}")

        columns: dict[str, list[float]] = {name: [] for name in REQUIRED_COLUMNS}
        for row_number, row in enumerate(reader, start=2):
            for name in REQUIRED_COLUMNS:
                try:
                    columns[name].append(float(row[name]))
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid numeric value in column {name!r}, row {row_number}."
                    ) from error

    if not columns["care_plan_cost_inr_thousands"]:
        raise ValueError(f"{path} contains no data rows.")

    arrays = {name: np.asarray(values, dtype=float) for name, values in columns.items()}
    if any(not np.all(np.isfinite(values)) for values in arrays.values()):
        raise ValueError(f"{path} contains non-finite numeric values.")
    if any(values.size != arrays["care_plan_cost_inr_thousands"].size for values in arrays.values()):
        raise ValueError(f"{path} contains rows with missing numeric values.")
    if np.any(arrays["care_plan_cost_inr_thousands"] <= 0):
        raise ValueError("care_plan_cost_inr_thousands must be positive for log transforms.")
    if np.any(arrays["avg_bill_inr_thousands"] <= 0):
        raise ValueError("avg_bill_inr_thousands must be positive for log transforms.")

    return arrays


def regression_metrics(
    y: np.ndarray, fitted: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    residuals = y - fitted
    tss = float(np.sum((y - np.mean(y)) ** 2))
    ess = float(np.sum((fitted - np.mean(y)) ** 2))
    rss = float(np.sum(residuals**2))
    r2 = 1.0 - rss / tss
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    return tss, ess, rss, r2, rmse, mae


def adjusted_r2(r2: float, n: int, predictor_count: int) -> float:
    return 1.0 - (1.0 - r2) * (n - 1) / (n - predictor_count - 1)


def main() -> None:
    records = load_records(DATA_PATH)
    value = records["care_plan_cost_inr_thousands"]
    x4 = records["avg_bill_inr_thousands"]
    n = value.size

    mean_value = float(np.mean(value))
    sd_value = float(np.std(value, ddof=1))
    se_value = sd_value / np.sqrt(n)
    ci95_low = mean_value - 1.96 * se_value
    ci95_high = mean_value + 1.96 * se_value

    z_stat = (mean_value - 8.0) / se_value
    reject_h0 = "yes" if abs(z_stat) > 1.96 else "no"
    n_required = int(np.ceil((1.96 * sd_value / 0.5) ** 2))

    mean_x4 = float(np.mean(x4))
    centered_x4 = x4 - mean_x4
    reg_slope = float(np.sum(centered_x4 * (value - mean_value)) / np.sum(centered_x4**2))
    reg_intercept = mean_value - reg_slope * mean_x4
    fitted = reg_intercept + reg_slope * x4
    tss, ess, rss, r2, rmse, mae = regression_metrics(value, fitted)
    tss_equals_ess_plus_rss = (
        "yes" if abs(tss - ess - rss) < 1e-6 * tss else "no"
    )

    x4_sd = float(np.std(x4))
    if x4_sd == 0:
        raise ValueError("avg_bill_inr_thousands has zero variance.")
    z_x4 = (x4 - mean_x4) / x4_sd
    gd_m = 0.0
    gd_c = 0.0
    gd_costs = {0: float(np.mean(value**2))}
    for iteration in range(1, 501):
        errors = gd_m * z_x4 + gd_c - value
        gd_m -= 0.1 * (2.0 / n) * float(np.sum(errors * z_x4))
        gd_c -= 0.1 * (2.0 / n) * float(np.sum(errors))
        if iteration in (1, 10, 100, 500):
            gd_costs[iteration] = float(np.mean((gd_m * z_x4 + gd_c - value) ** 2))

    gd_slope = gd_m / x4_sd
    gd_intercept = gd_c - gd_m * mean_x4 / x4_sd
    gd_slope_gap = abs(gd_slope - reg_slope)

    sorted_indices = np.argsort(fitted, kind="stable")
    lower_indices = sorted_indices[: n // 2]
    upper_indices = sorted_indices[n // 2 :]
    raw_residuals = value - fitted
    resid_sd_lower_raw = float(np.std(raw_residuals[lower_indices], ddof=1))
    resid_sd_upper_raw = float(np.std(raw_residuals[upper_indices], ddof=1))
    resid_sd_ratio_raw = resid_sd_upper_raw / resid_sd_lower_raw

    log_value = np.log(value)
    log_x4 = np.log(x4)
    mean_log_value = float(np.mean(log_value))
    centered_log_x4 = log_x4 - np.mean(log_x4)
    loglog_slope = float(
        np.sum(centered_log_x4 * (log_value - mean_log_value))
        / np.sum(centered_log_x4**2)
    )
    loglog_intercept = mean_log_value - loglog_slope * float(np.mean(log_x4))
    log_fitted = loglog_intercept + loglog_slope * log_x4
    _, _, _, loglog_r2, loglog_rmse, _ = regression_metrics(log_value, log_fitted)
    log_residuals = log_value - log_fitted
    resid_sd_lower_log = float(np.std(log_residuals[lower_indices], ddof=1))
    resid_sd_upper_log = float(np.std(log_residuals[upper_indices], ddof=1))
    resid_sd_ratio_log = resid_sd_upper_log / resid_sd_lower_log

    design = np.column_stack(
        (
            np.ones(n),
            log_x4,
            records["months_registered"],
            records["missed_visit_count_12m"],
            records["copay_burden_ratio"],
            records["adherence_score"],
            records["chronic_condition_flag"],
            records["insured_flag"],
        )
    )
    mr_beta = np.linalg.inv(design.T @ design) @ design.T @ log_value
    mr_fitted = design @ mr_beta
    _, _, _, mr_r2, _, _ = regression_metrics(log_value, mr_fitted)
    mr_adj_r2 = adjusted_r2(mr_r2, n, 7)
    loglog_adj_r2 = adjusted_r2(loglog_r2, n, 1)

    results: dict[str, int | float | str] = {
        "n": int(n),
        "mean_value": mean_value,
        "sd_value": sd_value,
        "se_value": se_value,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "z_stat": float(z_stat),
        "reject_h0": reject_h0,
        "n_required": n_required,
        "reg_slope": reg_slope,
        "reg_intercept": reg_intercept,
        "tss": tss,
        "ess": ess,
        "rss": rss,
        "tss_equals_ess_plus_rss": tss_equals_ess_plus_rss,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "gd_J0": gd_costs[0],
        "gd_J1": gd_costs[1],
        "gd_J10": gd_costs[10],
        "gd_J100": gd_costs[100],
        "gd_J500": gd_costs[500],
        "gd_slope": float(gd_slope),
        "gd_intercept": float(gd_intercept),
        "gd_slope_gap": float(gd_slope_gap),
        "resid_sd_lower_raw": resid_sd_lower_raw,
        "resid_sd_upper_raw": resid_sd_upper_raw,
        "resid_sd_ratio_raw": resid_sd_ratio_raw,
        "loglog_intercept": float(loglog_intercept),
        "loglog_slope": float(loglog_slope),
        "loglog_r2": float(loglog_r2),
        "loglog_rmse": float(loglog_rmse),
        "resid_sd_lower_log": resid_sd_lower_log,
        "resid_sd_upper_log": resid_sd_upper_log,
        "resid_sd_ratio_log": resid_sd_ratio_log,
        "mr_beta_0": float(mr_beta[0]),
        "mr_beta_1": float(mr_beta[1]),
        "mr_beta_2": float(mr_beta[2]),
        "mr_beta_3": float(mr_beta[3]),
        "mr_beta_4": float(mr_beta[4]),
        "mr_beta_5": float(mr_beta[5]),
        "mr_beta_6": float(mr_beta[6]),
        "mr_beta_7": float(mr_beta[7]),
        "mr_r2": float(mr_r2),
        "mr_adj_r2": float(mr_adj_r2),
        "loglog_adj_r2": float(loglog_adj_r2),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as output_file:
        for key, result in results.items():
            if isinstance(result, (int, np.integer)):
                formatted = str(int(result))
            elif isinstance(result, (float, np.floating)):
                formatted = f"{result:.4f}"
            else:
                formatted = result
            output_file.write(f"{key} = {formatted}\n")


if __name__ == "__main__":
    main()