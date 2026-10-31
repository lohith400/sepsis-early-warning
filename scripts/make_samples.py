"""Write the three illustrative synthetic patients as PhysioNet-format .psv files into samples/."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from sepsis.samples import SAMPLES, to_psv  # noqa: E402

out = os.path.join(os.path.dirname(__file__), "..", "samples")
os.makedirs(out, exist_ok=True)
names = {"stable": "patient_stable", "deteriorating": "patient_deteriorating", "shock": "patient_septic_shock"}
for k in SAMPLES:
    with open(os.path.join(out, names[k] + ".psv"), "w", encoding="utf-8") as fh:
        fh.write(to_psv(k))
    print("wrote", names[k] + ".psv")
