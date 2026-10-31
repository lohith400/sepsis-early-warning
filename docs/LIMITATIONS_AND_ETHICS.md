# Limitations and ethics

## Intended use
Teaching and research on soft computing (fuzzy systems, genetic algorithms, swarm optimization). The software is **not** a medical device, has not been clinically validated, and must not inform patient care.

## Known limitations
- **Many false alarms**: about 30 per true alert in the reference run; alert fatigue is a real clinical harm.
- **Time proxy**: the reference model selected ICU hours; risk therefore partly reflects length of stay.
- **Dataset scope**: PhysioNet 2019 Set A (US hospitals). No external validation; performance in other populations, devices or practices is unknown.
- **Label definition**: `SepsisLabel` marks a window defined by a clinical rule, not a confirmed diagnosis for every hour.
- **Missing data**: forward-fill and median imputation assume that missing means "unchanged / typical", which is not always true.
- **Fairness**: subgroup performance (age, sex, unit) was not analysed.
- **Example inputs**: the example patients on the monitor page are synthetic illustrations for trying the interface; they are not training data and not a model.

## Data and privacy
The training data are de-identified and publicly released by PhysioNet; follow the terms on the dataset page. The example patients shipped in this repository are **synthetic**. Uploaded `.psv` files are processed in memory and are not stored.

## Responsible reuse
Do not deploy this model in a clinical workflow. Any real use would need prospective validation, subgroup and calibration analysis, an alerting design that limits fatigue, and regulatory review.
