"""Synthetic records for the capstone (all values are invented; no real company data).
Run once from the repository root:  python generate_data.py
It writes data/records.csv and prints:  <rows> <columns> <positive_count> <md5 of the file>
Do not edit anything below the line marked NUMERIC CORE: every number in the brief was
measured on this exact output."""
import csv
import hashlib
import math
import os
import random

# ---------------- DOMAIN SETTINGS (names only; they never touch the numbers) ----------------
DOMAIN = "Healthcare"
ID_PREFIX = "H"
COLUMNS = [
    "record_id",
    "months_registered",
    "missed_visit_count_12m",
    "copay_burden_ratio",
    "avg_bill_inr_thousands",
    "adherence_score",
    "chronic_condition_flag",
    "insured_flag",
    "no_show",
    "care_plan_cost_inr_thousands"
]
# ---------------- NUMERIC CORE (identical in every domain) ----------------
N = 800
SEED = 2508
GROUP_P = [0.40, 0.35, 0.25]          # three hidden groups
MU_C1 = [48.0, 30.0, 14.0]            # tenure in months
MU_C2 = [2.0, 6.0, 4.0]               # count of friction events
MU_C3 = [0.22, 0.45, 0.72]            # ratio in [0, 1]
MU_C4 = [math.log(12.0), math.log(26.0), math.log(15.0)]   # log of an amount
MU_C5 = [62.0, 55.0, 40.0]            # score
P_B1 = [0.30, 0.55, 0.45]
P_B2 = [0.60, 0.50, 0.40]


def normal(rng):
    """One standard-normal draw from two uniform draws (Box-Muller)."""
    u1 = 1.0 - rng.random()
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def make_rows():
    rng = random.Random(SEED)
    rows = []
    for i in range(N):
        u = rng.random()
        g = 0 if u < GROUP_P[0] else (1 if u < GROUP_P[0] + GROUP_P[1] else 2)
        c1 = round(max(1.0, MU_C1[g] + 9.0 * normal(rng)), 1)
        c2 = int(max(0, round(MU_C2[g] + 1.6 * normal(rng))))
        c3 = round(min(0.99, max(0.01, MU_C3[g] + 0.12 * normal(rng))), 3)
        c4 = round(math.exp(MU_C4[g] + 0.35 * normal(rng)), 1)
        c5 = round(min(100.0, max(0.0, MU_C5[g] + 10.0 * normal(rng))), 1)
        b1 = 1 if rng.random() < P_B1[g] else 0
        b2 = 1 if rng.random() < P_B2[g] else 0
        eta = (-2.05 + 0.9 * (c3 - 0.45) / 0.25 - 0.7 * (c1 - 30.0) / 15.0
               + 0.35 * (c2 - 4.0) / 2.5 + 0.5 * b1 - 0.45 * b2 - 0.25 * (c5 - 55.0) / 12.0)
        p = 1.0 / (1.0 + math.exp(-eta))
        outcome = 1 if rng.random() < p else 0
        mu = 8.0 * math.exp(0.40 * (math.log(c4) - math.log(20.0)) / 0.35 + 0.25 * (c2 - 4.0) / 2.5
                            + 0.15 * b1 - 0.10 * (c1 - 30.0) / 15.0)
        value = round(mu * math.exp(0.45 * normal(rng) - 0.45 * 0.45 / 2.0), 2)
        rows.append([ID_PREFIX + format(i + 1, "04d"), c1, c2, c3, c4, c5, b1, b2, outcome, value])
    return rows


if __name__ == "__main__":
    data = make_rows()
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "records.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(data)
    with open(os.path.join("data", "records.csv"), "rb") as f:
        digest = hashlib.md5(f.read()).hexdigest()
    print(len(data), len(COLUMNS), sum(r[8] for r in data), digest)