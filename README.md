# Bearing Fault Diagnosis (Predictive Maintenance)

Classifies rotating-equipment bearing faults (Normal, Inner Race, Ball, Outer Race)
from vibration signals using FFT and time-domain features with Random Forest / XGBoost.
Built on the [CWRU Bearing Dataset](https://engineering.case.edu/bearingdatacenter).

## Pipeline
1. Load 40 CWRU 12 kHz drive-end recordings (4 loads x fault sizes x fault types)
2. **Split raw signals first** into train / validation / test (no window leakage)
3. Window signals (2048 samples, 50% overlap inside each split)
4. Extract features: RMS, kurtosis, skew, crest factor, 32 FFT band energies, spectral centroid/spread, dominant frequency
5. Fit scaler on train only, tune models on validation, evaluate once on test

### Split modes (`SPLIT_MODE` in `train.py`)
| Mode | Description |
|---|---|
| `segment` (default) | Each recording cut in time: 70% train / 15% val / 15% test, with a gap between segments |
| `file` | Whole recordings assigned to train / val / test (stricter, tests generalisation to unseen recordings) |

## Quick start
```bash
pip install -r requirements.txt
python download_data.py      # fetches data/cwru/*.mat
python train.py
```

### Google Colab
```python
!git clone https://github.com/<your-username>/bearing-fault-diagnosis.git
%cd bearing-fault-diagnosis
!pip install -q -r requirements.txt
!python download_data.py
!python train.py
```

## Outputs
`bearing_fault_model.joblib`, `confusion_matrix.png`, `feature_importance.png`, `fft_examples.png`
(written to the working directory; copy into `results/` for the repo if you want to show them).

## Dataset files
Normal: 97-100. Inner race: 105-108, 169-172, 209-212. Ball: 118-121, 185-188, 222-225.
Outer race (6 o'clock): 130-133, 197-200, 234-237. If automatic download fails, get them from
https://engineering.case.edu/bearingdatacenter/download-data-file and place them in `data/cwru/`.

## Results
_Add your test accuracy / macro-F1 and confusion matrix here after running._

## Roadmap
- Remaining useful life (RUL) regression on the NASA/IMS run-to-failure dataset
- Cross-load generalisation experiment (train on some loads, test on another)
