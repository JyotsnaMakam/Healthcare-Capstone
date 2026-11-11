"""Shared paths and formatting for the generated capstone files."""

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "records.csv"
OUTPUTS = ROOT / "outputs"


def load_data():
    """Return X1–X7, the binary outcome, and the cost, in file order."""
    with DATA_PATH.open(newline="", encoding="utf-8") as source:
        rows = csv.reader(source)
        next(rows)
        values = np.array([row[1:] for row in rows], dtype=float)
    return values[:, :7], values[:, 7].astype(int), values[:, 8]


def format_value(value):
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    return str(value)


def write_results(filename, results):
    OUTPUTS.mkdir(exist_ok=True)
    lines = [f"{key} = {format_value(value)}\n" for key, value in results.items()]
    (OUTPUTS / filename).write_text("".join(lines), encoding="utf-8")


def write_csv(filename, header, rows):
    OUTPUTS.mkdir(exist_ok=True)
    with (OUTPUTS / filename).open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(header)
        writer.writerows([format_value(value) for value in row] for row in rows)
