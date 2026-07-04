"""How noisy is NVIDIA's 100k-stratified eval? Train once (their raw recipe), then
score the FULL test set once and measure AP/AUC over many 100k stratified draws vs
the full 2.44M test. Shows the eval-sampling variance and the subsample-vs-full gap.
Runs in one GPU worker task (off the head node)."""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import ray

CSV = "/mnt/cluster_storage/transaction-fm/source/card_transaction.v1.csv"
FEATURE_COLS = ["User","Card","Year","Month","Day","Hour","Amount","Use Chip",
                "Merchant Name","Merchant City","Merchant State","Zip","MCC"]
XGB = {"n_estimators":400,"max_depth":8,"learning_rate":0.0023,"colsample_bytree":0.95,
       "min_child_weight":12,"subsample":0.673,"reg_alpha":0.01,"reg_lambda":0.001,"random_state":42}
N_DRAWS = 30


@ray.remote(num_gpus=1, num_cpus=6, memory=48 * 1024 ** 3)
def run():
    import numpy as np, pandas as pd
    from sklearn.compose import make_column_selector, make_column_transformer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.metrics import average_precision_score, roc_auc_score
    import xgboost as xgb
    np.random.seed(42)
    df = pd.read_csv(CSV); df.columns=[c.strip() for c in df.columns]
    date = pd.to_datetime(df["Year"].astype(str)+"-"+df["Month"].astype(str).str.zfill(2)+"-"
                          +df["Day"].astype(str).str.zfill(2), format="%Y-%m-%d")
    def cutoff(r):
        dc=date.value_counts().sort_index().cumsum(); return dc.index[dc>=dc.iloc[-1]*r][0]
    tc, sc = cutoff(0.8), cutoff(0.9)
    df["Hour"]=df["Time"].str.split(":",n=1,expand=True)[0].astype(int)
    df["Amount"]=df["Amount"].str.replace("$","",regex=False).str.replace(",","").astype(float)
    df["_target"]=((df["Is Fraud?"]=="Yes")|(df["Is Fraud?"]=="1")).astype(int)
    train_df, test_df = df[date<tc], df[date>=sc]
    val_df = df[(date>=tc)&(date<sc)]

    f=train_df.index[train_df._target==1].to_numpy(); nn=train_df.index[train_df._target==0].to_numpy()
    nf=min(len(f),100_000); idx=np.concatenate([np.random.choice(f,nf,False),
        np.random.choice(nn,min(len(nn),1_000_000-nf),False)]); np.random.shuffle(idx)
    Xtr, ytr = train_df.loc[idx,FEATURE_COLS].reset_index(drop=True), train_df.loc[idx,"_target"].values
    _,Xva,_,yva = train_test_split(val_df[FEATURE_COLS],val_df._target,test_size=100_000,
                                   stratify=val_df._target,random_state=42)
    pre=make_column_transformer((OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1),
        make_column_selector(dtype_include=["object","category"])),remainder="passthrough")
    Xtr_e=pre.fit_transform(Xtr); Xva_e=pre.transform(Xva)
    clf=xgb.XGBClassifier(**XGB,scale_pos_weight=1.0,tree_method="hist",device="cuda",
                          early_stopping_rounds=20,eval_metric="auc")
    clf.fit(Xtr_e,ytr,eval_set=[(Xva_e,yva.values)],verbose=False)

    # score the FULL test once
    yte = test_df["_target"].to_numpy()
    Xte_e = pre.transform(test_df[FEATURE_COLS])
    pte = clf.predict_proba(Xte_e)[:,1]
    full_ap=average_precision_score(yte,pte); full_auc=roc_auc_score(yte,pte)
    print(f"[var] FULL test: n={len(yte):,} frauds={int(yte.sum()):,}  AUC={full_auc:.4f}  AP={full_ap:.4f}",flush=True)

    # many 100k stratified draws from the full test (reuse precomputed preds)
    aps, aucs = [], []
    for s in range(N_DRAWS):
        _,ii = train_test_split(np.arange(len(yte)),test_size=100_000,stratify=yte,random_state=s)
        aps.append(average_precision_score(yte[ii],pte[ii])); aucs.append(roc_auc_score(yte[ii],pte[ii]))
    aps=np.array(aps); aucs=np.array(aucs)
    print(f"[var] 100k stratified eval over {N_DRAWS} draws (~{int(yte.sum()/len(yte)*100000)} frauds each):",flush=True)
    print(f"[var]   AP : mean={aps.mean():.4f}  std={aps.std():.4f}  min={aps.min():.4f}  max={aps.max():.4f}",flush=True)
    print(f"[var]   AUC: mean={aucs.mean():.4f}  std={aucs.std():.4f}  min={aucs.min():.4f}  max={aucs.max():.4f}",flush=True)
    print(f"[var] NVIDIA reported AP 0.1238 (one draw); our first draw was 0.1248. Spread above is the noise.",flush=True)
    return {"full_ap":float(full_ap),"full_auc":float(full_auc),
            "sub_ap_mean":float(aps.mean()),"sub_ap_std":float(aps.std()),
            "sub_ap_min":float(aps.min()),"sub_ap_max":float(aps.max())}


if __name__=="__main__":
    ray.init(ignore_reinit_error=True); print(ray.get(run.remote()),flush=True)
