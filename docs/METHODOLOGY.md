# Methodology

## 1. Data
PhysioNet / Computing in Cardiology Challenge 2019, **Training Set A**: 20,336 ICU stays (1,790 with sepsis), 790,215 hourly records with 40 clinical variables. `SepsisLabel` is 1 from 6 hours before the clinical onset of sepsis.

## 2. Split (no leakage)
Patients (not rows) are split 60 / 20 / 20 into train / validation / test, stratified by whether the patient ever develops sepsis. Inside the training patients, a further 70 / 30 split gives a *fit* set (parameters are learned) and a *select* set (GA and PSO candidates are judged). Validation and test rows keep the **natural** sepsis rate (about 2 %). For speed, GA and PSO use a sub-sample of the fit and select sets in which sepsis rows are over-sampled (4,000 + 8,000 and 2,500 + 5,000 rows).

## 3. Features (37 candidates)
Computed per patient, **using only past values**:

| Group | Features |
|---|---|
| Last known values (forward-filled) | HR, O2Sat, Temp, SBP, MAP, DBP, Resp, WBC, Lactate, Creatinine, Platelets, BUN |
| Derived | Shock index = HR / SBP, Age, ICULOS (hours in ICU) |
| Test ordering | number of times each of the 5 labs has been measured so far |
| 6-hour trends | change (`_d6`) and mean (`_m6`) of HR, O2Sat, Temp, SBP, MAP, Resp; variability (`_s6`) of HR, Temp, SBP, Resp, MAP |

Remaining gaps are filled with training medians (trend gaps with 0); all features are standardised with training statistics and clipped to ±5.

## 4. Genetic Algorithm
- Chromosome: 37 bits (1 = feature used). Between 3 and 6 bits may be set (a repair step enforces it).
- Fitness: AUROC of a logistic regression trained on the *fit* set and evaluated on the *select* set, minus 0.004 per selected feature.
- Operators: tournament selection (size 3), uniform crossover (p = 0.8), bit-flip mutation (p = 0.04 per bit), elitism (2). Population 30, 25 generations.

## 5. ANFIS (Takagi-Sugeno)
For $R$ rules and $K$ selected inputs, rule $r$ fires with strength

$$ w_r(x)=\exp\Big(-\tfrac12\sum_{k=1}^{K}\big(\tfrac{x_k-c_{rk}}{\sigma_{rk}}\big)^2\Big),\qquad \bar w_r=\frac{w_r}{\sum_j w_j} $$

and the model output is

$$ y(x)=\sum_{r=1}^{R}\bar w_r\,(a_{r0}+a_{r1}x_1+\dots+a_{rK}x_K). $$

- **Rule discovery:** k-means on the fit rows gives initial centres $c_{rk}$ and widths $\sigma_{rk}$ (cluster standard deviations, clipped to 0.3-2.5).
- **Hybrid learning:** for any given $(c,\sigma)$ the coefficients $a$ are the exact **class-balanced ridge least-squares** solution; PSO only searches $(c,\sigma)$.
- **Risk:** $\text{risk}=\sigma_{\text{sigmoid}}(\alpha\,y+\beta)$ with $\alpha,\beta$ fitted on validation patients (Platt scaling).

## 6. Particle Swarm Optimization
- Particle = all centres and log-widths ($2RK$ numbers). 40 particles, 60 rounds.
- Velocity update: $v \leftarrow \omega v + 1.5\,r_1(p_{\text{best}}-x)+1.5\,r_2(g_{\text{best}}-x)$ with inertia $\omega$ decaying linearly from 0.9 to 0.4; velocities are clipped to 20 % of the search range.
- Initialisation: one particle = k-means solution, half the swarm near it, the rest uniform random.
- Loss (on the *select* patients): $(1-\mathrm{AUROC}) + 0.1\times$ class-weighted squared error.

## 7. Threshold and evaluation
- The alert threshold is the highest risk value whose **validation** sensitivity is at least 80 %.
- The test set is touched once. Reported: AUROC, AUPRC, sensitivity, specificity, precision, F1, accuracy, false alarms per true alert, and the share of sepsis / non-sepsis patients alerted at least once.
- **Baselines** (Logistic Regression, Random Forest on the GA features and on all 37 features) use the same split and the same threshold rule.
- Accuracy is reported but is not a meaningful headline measure at a 2 % event rate.

## 8. Sustained alert (extension)
The monitor page also flags the hour at which the risk has been above the threshold for 3 consecutive hours. It is a usability aid; its effect on false alarms has **not** been evaluated.
