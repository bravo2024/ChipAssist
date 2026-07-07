# ChipAssist

Predicts compiler optimisation outcomes for reconfigurable dataflow (RDU-style) architectures.

Trains four classifiers on synthetic compiler feature data to predict whether the SNN dataflow compiler will achieve speedup for a given hardware mapping configuration. Features include model parameters, sequence length, precision bits, sparsity, reconfigurable lanes, and placement efficiency.

## Setup

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## What's in the app

| Tab | What it does |
|---|---|
| **Explorer** | Dataset overview, speedup distribution, feature descriptions |
| **Model Lab** | Multi-model comparison table, ROC curves, calibration plots, CV results |
| **Compilation Analysis** | Dataflow compiler theory (latency model, PE utilization, placement efficiency), feature importance for XGBoost |
| **Hardware Mapping** | Interactive PE/tile configurator — adjust lanes, precision, batch size to see peak TOPS |

## How well it predicts speedup

Best model (Random Forest) holdout results:

| Metric | Value |
|---|---|
| ROC AUC | 0.630 |
| Gini | 0.259 |
| KS Statistic | 0.190 |
| F1 Score | 0.487 |
| Accuracy | 0.608 |

5-fold CV AUC: 0.638 ± 0.016, four models compared. Worth noting the AUC here is well short of the other repos in this set — the underlying speedup/no-speedup boundary in the synthetic compiler data is genuinely fuzzier than, say, a fraud or churn label.

## Under the hood

Synthetic dataset engineered to mirror RDU compiler inputs: model parameter count, sequence length, arithmetic precision (INT4/8/16), sparsity, reconfigurable lane count, batch size, memory bandwidth, and placement efficiency.

```
ChipAssist/
  src/         data, model, evaluate, persist modules
  train.py     training pipeline (multi-model + CV)
  app.py       Gradio dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

MIT licensed.
