Domain: Healthcare

# Healthcare-Capstone

## PharmEasy Healthcare Decision Intelligence — Capstone Project

## Project Context & Rules for AI Assistant
- **Goal:** Build an end-to-end operations-analytics pipeline across 4 parts for PharmEasy (IIT Mandi AI & Data Science Minor).
- **Tech Stack & Libraries:** Python 3.10+ and `numpy` only. No ready-made machine learning or stats packages (e.g., scikit-learn, statsmodels, scipy) may produce graded numbers. Use NumPy array arithmetic and `numpy.linalg` for linear algebra[cite: 2].
- **Dataset:** `data/records.csv` generated via `generate_data.py` (800 rows, 10 columns)[cite: 2].
- **Column Mappings:**
  - `record_id` (Index, unused in math)[cite: 2]
  - `months_registered` (X1)[cite: 2]
  - `missed_visit_count_12m` (X2)[cite: 2]
  - `copay_burden_ratio` (X3)[cite: 2]
  - `avg_bill_inr_thousands` (X4)[cite: 2]
  - `adherence_score` (X5)[cite: 2]
  - `chronic_condition_flag` (X6, 0/1)[cite: 2]
  - `insured_flag` (X7, 0/1)[cite: 2]
  - `no_show` (Outcome: 1 for missed visit, 0 otherwise)[cite: 2]
  - `care_plan_cost_inr_thousands` (Value target)[cite: 2]

## How to run
```bash
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
