Domain: Healthcare

## How to run

Run these commands in the repository root after cloning. Use Python 3.10 or newer and install the pinned NumPy version into the same environment.

```bash
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

The generator prints `800 10 170 e26b46e82f1a1af44dcabc133050802e`. The four scripts write the results to `outputs/`. All records are synthetic. PharmEasy is the scenario setting, and the analysis covers scheduling, cost planning, and routing.

## Part 1 findings

**R1.1 — Planning assumption.** Reject the INR 8,000 mean at the 5% significance level. The absolute z statistic of 2.8930 exceeds 1.96. A Type I error would be telling the planning team its assumed mean is wrong when the true mean really is INR 8,000. The estimated gap is 0.9382 thousand INR per patient. That is large enough to matter for the budget because it accumulates across patients.

**R1.2 — Sample size.** The required sample is 1,293 records versus 800 available. With this observed variability, the team needs 493 more records to target a 95% margin of error of INR 500.

**R1.3 — Gradient descent.** The slope gap is 0.0000 to four decimal places. Standardising the predictor gives well-scaled gradients, and 500 updates on the convex squared-error objective reach the same minimum as the closed-form solution.

**R1.4 — Residual spread.** The raw upper-to-lower residual SD ratio is 3.7525, which shows a clear funnel. After logging both variables, the ratio is 1.0396, so the two halves have nearly the same spread and the funnel is largely removed by this check.

**R1.5 — Multiple regression.** The coefficients above 0.05 in absolute value, excluding the intercept, are ln(average past bill) (1.1576), missed visits (0.0940), copay burden ratio (0.1067), and chronic condition flag (0.1518). This is a mechanical cutoff across coefficients measured in different units. Adjusted R² increases from 0.6561 for the single-predictor log model to 0.7403 for the multiple log model, so the added predictors improve in-sample fit after the predictor-count penalty.

## Part 2 findings

**R2.1 — Learning rates.** After 200 updates, the losses from lowest to highest are 0.3703 (η = 1.0), 0.3776 (η = 0.1), and 0.5091 (η = 0.01). The smallest rate makes slow progress in 200 updates, while 1.0 converges faster here, though an excessively large rate could overshoot the minimum.

**R2.2 — Predictors and odds.** By absolute weight, the top three predictors are copay burden ratio (0.9013), months registered (−0.5554), and missed visits in the last 12 months (0.5505). With other predictors fixed, a one-training-SD increase in copay burden multiplies the estimated odds of a missed appointment by 2.4629.

**R2.3 — Accuracy and thresholds.** Always predicting attendance gives accuracy 0.7875, close to the logistic model’s 0.8042 at τ = 0.5. Attendance is the majority class, so accuracy alone can hide missed no-shows. Lowering τ from 0.5 to 0.3 increases recall from 0.4706 to 0.7451 and lowers precision from 0.5455 to 0.4634.

**R2.4 — Logistic versus KNN.** Logistic regression at τ = 0.3 reaches F1 0.5714 and recall 0.7451. KNN with the cross-validated K of 5 reaches F1 0.3696 and recall 0.3333.

**R2.5 — Reminder costs.** I would use logistic regression at τ = 0.15. The test cost is INR 6,050 at that threshold, INR 11,800 at τ = 0.5, and INR 20,400 if nobody is flagged. Choosing τ on the training rows reserves the test rows for evaluating the resulting policy.

## Part 3 findings

**R3.1 — Elbow.** The successive WCSS drops are 659.8282, 108.0739, 171.0575, and 45.5466. The drop after K = 3 is less than half the drop before it, making K = 3 the first qualifying elbow. The drops do not shrink steadily because the pinned starting rows produce reproducible solutions that need not be global optima.

**R3.2 — Service profiles.**

| Segment | Service profile | Scheduling action |
|---|---|---|
| A | Established regulars. Longest average registration and fewest prior missed visits. | Use a routine self-service confirmation. |
| B | Frequent reschedulers. Most prior missed visits and highest average bill. | Offer a one-click rescheduling option in the confirmation message. |
| C | Newer members with high copays. Shortest average registration and highest copay burden. | Offer an early reminder with a clear copay estimate and a way to confirm the slot. |

Segment C has the highest observed no-show rate, 0.4664.

**R3.3 — Cohort cuts.** Single linkage gives {E1}, {E4}, {E5}, and {E2, E3, E6}. Complete linkage gives {E1, E4, E5} and {E2, E3, E6}. Single linkage uses the nearest pair, so a short cross-group connection compresses the later merge heights and the largest gap falls earlier. Complete linkage uses the farthest pair, leaving a large final gap when the two three-member groups are merged.

**R3.4 — PCA.** The five feature means are zero after standardisation, to numerical precision. Registration tenure (+0.5317) and adherence (+0.4418) contribute positively to PC1, while copay burden (−0.5187), missed visits (−0.4100), and average past bill (−0.2916) contribute negatively. Three components retain at least 80% of the variance (0.8616 cumulative). Mean PC1 scores are 1.7773 for A, −0.7831 for B, and −1.6709 for C, placing A on the longer-tenure, lower-friction side and C farthest in the opposite direction.

## Part 4 findings

**R4.1 — Split quality.** S4, registration tenure at most 24 months, reduces impurity the most of the four candidates, with information gain 0.1252 and Gini gain 0.0606. A tree grown to pure leaves can memorise noise in small groups and generalise poorly. A maximum depth stops growth before that happens.

**R4.2 — Model choice.** I would ship logistic regression, which has the highest F1 in the five-model comparison (0.5714 at recall 0.7451). The stump raises recall to 0.7647 but drops F1 to 0.5532. Hard voting lowers both F1 and recall to 0.5344 and 0.6863. Soft voting matches logistic recall at 0.7451 and lowers F1 to 0.5507, so combining models provides no improvement on these metrics.

**R4.3 — Boosting.** The two points misclassified in round 1 have a combined normalised weight of 0.5000. Round 2 trusts h2 more because its weighted error is lower (0.1667 versus 0.2500), giving it a larger alpha (0.8047 versus 0.5493).

**R4.4 — Architecture.** Vector layers are shown as 1 × 1 × depth to keep the output shape in the same format as the spatial layers.

| Model A CNN layer | Output shape | Parameters |
|---|---:|---:|
| Convolution 1 | 28 × 28 × 6 | 156 |
| ReLU 1 | 28 × 28 × 6 | 0 |
| Max-pool 1 | 14 × 14 × 6 | 0 |
| Convolution 2 | 10 × 10 × 16 | 2,416 |
| ReLU 2 | 10 × 10 × 16 | 0 |
| Max-pool 2 | 5 × 5 × 16 | 0 |
| Flatten | 1 × 1 × 400 | 0 |
| Dense 1 with ReLU | 1 × 1 × 32 | 12,832 |
| Dense output | 1 × 1 × 10 | 330 |

The CNN has 15,734 parameters and the MLP has 13,002 on the 28 × 28 input. At 128 × 128 × 3, the comparison dense layer has 50,332,672 parameters and the convolution has 1,792 because each filter is shared across positions.

For Model B, W_xh is 128 × 64, W_hh is 128 × 128, and W_hy is 16 × 128. The total stays at 26,768 parameters for longer notes because every time step reuses the same input and recurrent weights and hidden bias. For the 10-class image router, use ReLU in the hidden layers and softmax with categorical cross-entropy for the output. In a deep plain network, repeated multiplication by small derivatives can make the gradient too weak for early layers to learn. An identity skip connection gives the gradient a shorter path through the network.

## Model-choice memo

**To the head of scheduling operations**

Use logistic regression to flag appointments for reminders at a no-show probability of at least 0.15, the threshold selected on the training rows. Its held-out cost is INR 6,050, the Task 2.8 logistic comparator. The same model at 0.50 costs INR 11,800, and flagging nobody costs INR 20,400.

In the five-model comparison at the specified thresholds, logistic regression achieved F1 0.5714 and recall 0.7451 at 0.30, ahead of both voting models on F1. Those comparison scores are for 0.30. The 0.15 recommendation follows the separate training-set cost rule, which gives missed no-shows more weight than false alarms.

The evidence comes from synthetic records, so the measured costs do not establish what would happen in actual operations. Before expanding reminder coverage, validate alert volumes and costs in a monitored scheduling pilot. Defer both the scan router and the referral-note classifier until the structured reminder model is in production.

## Limitations

- Synthetic relationships may differ from real scheduling patterns, so neither the model scores nor the segment profiles establish real-world performance.
- Evaluation uses one fixed split with 51 test positives. Recall and cost can change substantially with a different set of appointments.
- The cost comparison assumes INR 50 per false alarm and INR 400 per missed no-show. It does not measure whether reminders prevent no-shows or how many calls staff can make.
- The cost regressions describe in-sample fit on these records. Their adjusted R² and residual patterns do not measure out-of-sample forecasting performance.
