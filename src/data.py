from __future__ import annotations
import numpy as np; import pandas as pd
FEATURE_NAMES = ["model_parameters_billions","sequence_length","precision_bits","sparsity_pct","reconfig_lanes","batch_size","memory_bandwidth_gbps","placement_efficiency","dataflow_ops","compiler_iterations"]
CATEGORICAL_FEATURES = ["precision_bits"]
NUMERICAL_FEATURES = ["model_parameters_billions","sequence_length","sparsity_pct","reconfig_lanes","batch_size","memory_bandwidth_gbps","placement_efficiency","dataflow_ops","compiler_iterations"]
TARGET_NAME = "speedup_achieved"
def make_synthetic(n=10000,seed=42):
    rng=np.random.default_rng(seed)
    df=pd.DataFrame({
        "model_parameters_billions": rng.lognormal(mean=0.5,sigma=1.0,size=n).clip(0.1,500).round(2),
        "sequence_length": rng.choice([128,256,512,1024,2048,4096,8192],size=n),
        "precision_bits": rng.choice([4,8,16],size=n,p=[0.15,0.55,0.30]),
        "sparsity_pct": rng.beta(2,5,size=n).round(3),
        "reconfig_lanes": rng.choice([32,64,128,256],size=n,p=[0.10,0.30,0.40,0.20]),
        "batch_size": rng.choice([1,2,4,8,16,32,64],size=n),
        "memory_bandwidth_gbps": rng.uniform(100,2000,size=n).round(1),
        "placement_efficiency": rng.beta(6,3,size=n).round(3),
        "dataflow_ops": rng.lognormal(mean=10,sigma=1.5,size=n).clip(1e2,1e7).round(0),
        "compiler_iterations": rng.poisson(lam=50,size=n).clip(5,500),
    })
    params=np.clip(np.log(df["model_parameters_billions"]+1)/6,0,1)
    seq=np.clip(df["sequence_length"]/8192,0,1)
    prec_map={4:1.0,8:0.6,16:0.3}; prec=np.array([prec_map[p] for p in df["precision_bits"]])
    sparsity=df["sparsity_pct"]; lanes=df["reconfig_lanes"]/256
    batch=np.clip(df["batch_size"]/64,0,1); mem=df["memory_bandwidth_gbps"]/2000
    placement=df["placement_efficiency"]; ops=np.log(df["dataflow_ops"]+1)/16
    comp=np.clip(df["compiler_iterations"]/500,0,1)
    log_odds = 0.5 + 0.3*params + 0.2*seq + 0.5*prec + 0.3*sparsity + 0.2*lanes + 0.1*batch + 0.2*mem + 0.4*placement + 0.1*ops + 0.3*comp + rng.normal(0,0.5,size=n)
    prob=1/(1+np.exp(-log_odds)); y=(prob>np.percentile(prob,65)).astype(np.float64)
    return {"X":df,"y":y,"features":FEATURE_NAMES,"df":df.assign(speedup_achieved=y),"categorical_features":CATEGORICAL_FEATURES,"numerical_features":NUMERICAL_FEATURES,"n_samples":n,"n_features":len(FEATURE_NAMES),"positive_rate":y.mean()}
