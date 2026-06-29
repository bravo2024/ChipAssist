from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from src.data import make_synthetic, TARGET_NAME
from src.model import train_all_models, cross_validate
from src.visualizations import *
st.set_page_config(page_title="ChipAssist | SambaNova RDU Compiler", layout="wide", page_icon="\U0001f9ea")
with st.sidebar:
    st.header("\u2699 Config"); n=st.slider("Samples",2000,20000,10000,1000); tau=st.slider("Threshold",0.05,0.95,0.50,0.05)
    st.caption("SambaNova | RDU Dataflow Compiler Optimization")
data=make_synthetic(n=n); b=train_all_models(data)
y_test=b["y_test"]; y_probas={n:b["results"][n]["y_proba"] for n in b["results"]}
best=max(b["results"],key=lambda n: b["results"][n]["metrics"].get("roc_auc",0))
c1,c2,c3,c4=st.columns(4)
c1.metric("Samples",f"{n:,}"); c2.metric("Speedup Rate",f"{data['positive_rate']:.1%}")
c3.metric("Best AUC",f"{b['results'][best]['metrics']['roc_auc']:.4f}"); c4.metric("Best",best)
t1,t2,t3,t4=st.tabs(["\U0001f4ca Explorer","\U0001f52c Model Lab","\U0001f4a1 Compilation Analysis","\U0001f527 Hardware Mapping"])
with t1:
    st.dataframe(data["df"].head(50),use_container_width=True,height=200)
    fig,ax=plt.subplots(figsize=(5,3)); _style()
    ax.bar(["No Speedup","Speedup"],[1-data["positive_rate"],data["positive_rate"]],color=["#f43f5e","#22c55e"])
    for i,v in enumerate([1-data["positive_rate"],data["positive_rate"]]): ax.text(i,v+.01,f"{v:.1%}",ha="center",color="white")
    ax.set_title("RDU Compilation Speedup Distribution",color="white"); ax.grid(True,alpha=.2)
    st.pyplot(fig)
    st.markdown("**SambaNova SNN compiler inputs:** model_parameters_billions, sequence_length, precision_bits (4/8/16), sparsity_pct, reconfig_lanes, batch_size, memory_bandwidth_gbps, placement_efficiency, dataflow_ops, compiler_iterations")
with t2:
    rows=[{**{"Model":n},**{k:f"{v:.4f}" for k,v in r["metrics"].items() if k!="confusion_matrix"}} for n,r in b["results"].items()]
    st.dataframe(pd.DataFrame(rows).set_index("Model"),use_container_width=True)
    col_a,col_b=st.columns(2)
    with col_a: st.pyplot(plot_roc_curve(y_test,y_probas))
    with col_b: st.pyplot(plot_calibration_curve(y_test,y_probas))
    st.pyplot(plot_confusion_matrix(y_test,b["results"]["XGBoost"]["y_pred"],"XGBoost"))
    cv=cross_validate(data); cvr=[{"Model":n,"AUC":f"{s['roc_auc']['mean']:.4f}","\u00b1":f"\u00b1{s['roc_auc']['std']:.4f}"} for n,s in cv.items()]
    st.dataframe(pd.DataFrame(cvr).set_index("Model"),use_container_width=True)
with t3:
    st.subheader("Dataflow Compilation Analysis")
    st.latex(r"\text{Speedup} = \frac{T_{\text{baseline}}}{T_{\text{RDU}}}, \quad T_{\text{RDU}} = \max\!\left(T_{\text{compute}}, T_{\text{memory}}, T_{\text{interconnect}}\right)")
    st.caption("SambaNova's RDU uses a reconfigurable dataflow architecture where the compiler maps ML graph nodes to a 2D systolic array of PEs. Speedup is limited by the critical path through compute, memory, or interconnect resources.")
    st.latex(r"\text{RDU Utilization} = \frac{\sum_{\text{tiles}} \text{Active PE}_{\text{tile}}}{\text{Total PE}} \times 100\%")
    st.caption("PE (Processing Element) utilization measures how efficiently the dataflow graph maps onto the RDU's 2D tile grid. Higher utilization indicates better compiler placement and routing.")
    st.latex(r"\text{Placement Efficiency} = \frac{\text{Optimal Cycles}}{\text{Actual Cycles}} = \frac{\sum w_i \cdot d_i}{\sum w_i \cdot (d_i + s_i)}")
    st.caption("Compiler placement quality: ratio of ideal to actual communication distance between dependent operations. SambaNova's SNN compiler uses ILP and heuristic-driven placement to minimize data movement latency.")
    importances=b["models"]["XGBoost"].feature_importances_
    fi=pd.DataFrame({"feature":data["features"],"importance":importances}).sort_values("importance",ascending=True)
    fig,ax=plt.subplots(figsize=(6,6)); _style()
    ax.barh(fi["feature"],fi["importance"],color="#22d3ee")
    ax.set_title("Speedup Driver Importance — SNN Compiler",color="white")
    ax.set_xlabel("Importance"); ax.grid(True,alpha=.2)
    st.pyplot(fig)
with t4:
    st.subheader("Hardware Mapping — RDU Tile Configuration")
    st.latex(r"N_{\text{PE}} = \text{ReconfigLanes}^2 \times \text{BatchSize}, \quad \text{TOPS}_{\text{peak}} = N_{\text{PE}} \times f_{\text{clk}} \times 2")
    st.caption("RDU peak throughput: each tile contains a grid of PEs. At 1 GHz with 256 lanes and batch 1, a single RDU achieves ~512 TOPS (INT8). The compiler selects optimal lane × batch tiling.")
    lane_choice=st.selectbox("Reconfigurable Lanes",[32,64,128,256],index=2)
    prec_choice=st.selectbox("Precision (bits)",[4,8,16],index=1)
    batch_choice=st.selectbox("Batch Size",[1,2,4,8,16,32,64],index=2)
    n_pe=lane_choice**2*batch_choice
    tops=n_pe*1e9*2/(1e12*{4:0.5,8:1.0,16:2.0}[prec_choice])
    st.metric("Active PEs",f"{n_pe:,}")
    st.metric("Peak TOPS",f"{tops:.1f}")
    st.caption("SambaNova's Cardinal SN30 RDU delivers up to 2,048 PEs per system node with 80 GB HBM2e memory bandwidth, supporting models up to 500B parameters in a multi-node configuration.")
