# ===================================================================================
#  EXPORT THE TRAINED MODEL  (paste as a NEW LAST CELL of the Colab notebook and run it)
#  Run it in the SAME session, after the notebook has finished (all cells executed).
#  It saves model.json, which the web app loads from  artifacts/model.json
#  Nothing is retrained: it only writes out what the notebook already computed.
# ===================================================================================
import json, datetime
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

EXPECTED_FEATURES = ['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','WBC','Lactate','Creatinine','Platelets','BUN',
    'ShockIndex','Age','ICULOS','WBC_count','Lactate_count','Creatinine_count','Platelets_count','BUN_count',
    'HR_d6','HR_m6','O2Sat_d6','O2Sat_m6','Temp_d6','Temp_m6','SBP_d6','SBP_m6','MAP_d6','MAP_m6','Resp_d6','Resp_m6',
    'HR_s6','Temp_s6','SBP_s6','Resp_s6','MAP_s6']
assert list(FEATS) == EXPECTED_FEATURES, "Feature list differs from the web app's list. Use the v2 notebook."
for _v in ['gbest','th1','platt','mu','sd','sel_cols','sel_names','thr_anfis','results','roc_data','pso_hist','ga_best_hist']:
    assert _v in globals(), f"Variable '{_v}' not found. Run ALL notebook cells first (same session), then this cell."

_f = float
def _curves(yy, s, n=200):
    fpr, tpr, _ = roc_curve(yy, s); i = np.unique(np.linspace(0, len(fpr)-1, min(n, len(fpr))).astype(int))
    p, r, _ = precision_recall_curve(yy, s); j = np.unique(np.linspace(0, len(p)-1, min(n, len(p))).astype(int))
    return ({'fpr': [round(_f(v),5) for v in fpr[i]], 'tpr': [round(_f(v),5) for v in tpr[i]]},
            {'recall': [round(_f(v),5) for v in r[j]], 'precision': [round(_f(v),5) for v in p[j]]})
def _model_row(r):
    return {'model': r['Model'], 'auroc': _f(r['AUROC']), 'auprc': _f(r['AUPRC']), 'sensitivity': _f(r['Sensitivity']),
            'specificity': _f(r['Specificity']), 'precision': _f(r['Precision']), 'f1': _f(r['F1']), 'accuracy': _f(r['Accuracy']),
            'false_alarms_per_true_alert': _f(r['FalseAlarms_per_TrueAlert']),
            'sepsis_patients_alerted': _f(r['SepsisPatients_alerted']), 'non_sepsis_patients_alerted': _f(r['NonSepsisPatients_alerted']),
            'train_time_s': _f(r.get('TrainTime_s', 0.0))}

R_, K_ = N_RULES, len(sel_cols)
C_ = np.asarray(gbest[:R_*K_]).reshape(R_, K_); S_ = np.exp(np.asarray(gbest[R_*K_:])).reshape(R_, K_)
roc_all, pr_all = {}, {}
for name, (yy, ss) in roc_data.items():
    roc_all[name], pr_all[name] = _curves(yy, ss)
trF = F[tr_mask]
medians_ = trF.median()          # imputation values (median of the filled data equals the original median)
models_ = [_model_row(r) for r in results]

artifact = {
    'format_version': 1,
    'name': 'GA + ANFIS + PSO sepsis early-warning model (trained in the Colab notebook)',
    'version': '2.0.0',
    'source': 'colab-notebook-v2',
    'created': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    'config': {'seed': SEED, 'target_sens': TARGET_SENS, 'k_min': K_MIN, 'k_max': K_MAX, 'n_rules': N_RULES,
               'ga_pop': GA_POP, 'ga_gen': GA_GEN, 'pso_particles': PSO_PARTICLES, 'pso_iters': PSO_ITERS,
               'rf_trees': 200, 'fit_pos': 4000, 'fit_neg': 8000, 'sel_pos': 2500, 'sel_neg': 5000, 'n_patients': N_PATIENTS},
    'dataset': {'patients': int(len(pid_all)), 'records': int(len(df)), 'sepsis_patients': int(pid_all.sum()),
                'sepsis_records': int(y.sum()), 'sepsis_rate': _f(y.mean()), 'train_patients': int(len(tr_p)),
                'validation_patients': int(len(va_p)), 'test_patients': int(len(te_p)),
                'validation_rows': int(len(va_idx)), 'test_rows': int(len(te_idx)), 'test_sepsis_rate': _f(y[te_idx].mean()),
                'fit_rows': int(len(fit_idx)), 'select_rows': int(len(sel_idx))},
    'features': {'all': list(FEATS), 'selected': list(sel_names), 'selected_idx': [int(i) for i in sel_cols]},
    'preprocessing': {'clip': 5.0,
                      'medians': {k: _f(v) for k, v in medians_.items()},
                      'mean': {k: _f(v) for k, v in mu.items()},
                      'std': {k: _f(v) for k, v in sd.items()},
                      'range_min': {n: _f(trF[n].min()) for n in sel_names},
                      'range_max': {n: _f(trF[n].max()) for n in sel_names}},
    'anfis': {'n_rules': int(R_), 'centers': C_.tolist(), 'sigmas': S_.tolist(), 'theta': np.asarray(th1).tolist()},
    'calibration': {'coef': _f(platt.coef_[0][0]), 'intercept': _f(platt.intercept_[0])},
    'alert': {'threshold': _f(thr_anfis), 'target_sensitivity': _f(TARGET_SENS)},
    'metrics': {'anfis': models_[0], 'confusion': {'tn': int(cm[0]), 'fp': int(cm[1]), 'fn': int(cm[2]), 'tp': int(cm[3])},
                'models': models_, 'ga_quick_auc_selected': _f(ga_fitness(best)[1]),
                'ga_quick_auc_all': _f(auc_fast(yb, lr_all.decision_function(Xb))),
                'val_auc_kmeans_only': _f(auc_v0), 'val_auc_after_pso': _f(auc_v1)},
    'curves': {'roc': roc_all, 'pr': pr_all, 'test_prevalence': _f(y[te_idx].mean())},
    'history': {'ga_best': [_f(v) for v in ga_best_hist], 'ga_mean': [_f(v) for v in ga_mean_hist],
                'pso': [_f(v) for v in pso_hist], 'pso_start': _f(init_fit)},
    'timing': {'ga_s': _f(ga_time), 'pso_s': _f(pso_time), 'predict_us_per_record': _f(per_rec_us)},
}
with open('model.json', 'w') as fh:
    json.dump(artifact, fh)
print('Saved model.json')
print('  selected features :', list(sel_names))
print('  test AUROC        : %.4f (the report says 0.7547 for the original run)' % artifact['metrics']['anfis']['auroc'])
print('  alert threshold   : %.4f' % artifact['alert']['threshold'])
try:
    from google.colab import files
    files.download('model.json')
except Exception as e:
    print('Download manually from the Files panel (folder icon on the left):', e)
