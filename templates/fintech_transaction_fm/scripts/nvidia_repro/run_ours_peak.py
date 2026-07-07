"""OPEN #1 — seed x eval bootstrap on OUR OWN Ray-trained FM embeddings (nvours_embed_*.npy,
saved by run_ours_full.py). Identical method to run_peakhunt.py (which used NVIDIA's weights);
only the embedding source + output path change. Reports the PEAK fusion AP and the fraction of
100K eval draws >= 0.1755 — the same favorable-single-draw basis NVIDIA's published 0.1755 uses —
plus each seed's full-eval (typical) fusion AP and raw on the same resamples, for an honest read.
Our fm is ~2x NVIDIA's, so fusion should clear 0.176 more often than the their-weights run did.
"""
import ray, json
ray.init(ignore_reinit_error=True)

@ray.remote(num_gpus=1, num_cpus=8)
def go():
    import sys, numpy as np, pandas as pd
    sys.path.insert(0, ".")
    import cuml.preprocessing  # shim
    import cudf
    from src.tokenizer import FinancialTokenizerPipeline
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.compose import make_column_transformer, make_column_selector
    from sklearn.decomposition import PCA
    from sklearn.metrics import average_precision_score
    import xgboost as xgb
    SD="/mnt/cluster_storage/nvidia_data/temporal_split"
    FC=['User','Card','Year','Month','Day','Hour','Amount','Use Chip','Merchant Name','Merchant City','Merchant State','Zip','MCC']
    HASH,BAL=2000,1_000_000; pq={"train":"train.parquet","val":"val_eval.parquet","test":"test_eval.parquet"}
    emb,y,raw={},{},{}
    for split in ("train","val","test"):
        e=np.load(f"/mnt/cluster_storage/nvours_embed_{split}.npy")   # OUR weights
        gdf=cudf.read_parquet(f"{SD}/{pq[split]}"); fr=gdf["Is Fraud?"]
        lbl=((fr=="Yes")|(fr=="1")).astype("int32").to_pandas().to_numpy()
        if split=="train":
            f=np.where(lbl==1)[0].tolist(); nrm=np.where(lbl==0)[0].tolist()
            np.random.seed(42); nf=min(len(f),int(BAL*0.1)); nn=min(len(nrm),BAL-nf)
            s=np.concatenate([np.random.choice(f,nf,replace=False),np.random.choice(nrm,nn,replace=False)])
            np.random.shuffle(s); gdf=gdf.iloc[s].reset_index(drop=True); lbl=lbl[s]
        base=gdf.to_pandas()
        base['Hour']=base['Time'].str.split(':',n=1,expand=True)[0].astype(int)
        base['Amount']=base['Amount'].str.replace('$','',regex=False).str.replace(',','').astype(float)
        base=base.reset_index(drop=True)
        gdf2=gdf.copy(); gdf2["__row_id__"]=np.arange(len(gdf2),dtype=np.int64)
        pip=FinancialTokenizerPipeline(merchant_hash_size=HASH); gdf2=pip.preprocess(gdf2)
        rid=gdf2["__row_id__"].to_pandas().to_numpy(np.int64)
        raw[split]=base.loc[rid,FC].reset_index(drop=True); emb[split]=e; y[split]=lbl[rid]
        assert len(e)==len(rid), f"{split}: emb {len(e)} != rid {len(rid)} — order mismatch"
    pca=PCA(n_components=64,random_state=42)
    Xe={k:(pca.fit_transform(emb["train"]) if k=="train" else pca.transform(emb[k])) for k in ("train","val","test")}
    pre=make_column_transformer((OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),make_column_selector(dtype_include=['object','category'])),remainder='passthrough')
    Xr={"train":pre.fit_transform(raw["train"]),"val":pre.transform(raw["val"]),"test":pre.transform(raw["test"])}
    Ctr=np.hstack([Xr["train"],Xe["train"]]);Cval=np.hstack([Xr["val"],Xe["val"]]);Cte=np.hstack([Xr["test"],Xe["test"]])
    P=dict(n_estimators=512,max_depth=12,learning_rate=0.00305,colsample_bytree=0.768,min_child_weight=25.85,subsample=0.65,reg_alpha=0.01,reg_lambda=0.0001,gamma=4.8)
    praw=dict(n_estimators=400,max_depth=8,learning_rate=0.0023,colsample_bytree=0.95,min_child_weight=12,subsample=0.673,reg_alpha=0.01,reg_lambda=0.001,random_state=42)
    rc=xgb.XGBClassifier(**praw,scale_pos_weight=1.0,tree_method='hist',device='cuda',early_stopping_rounds=20,eval_metric='auc')
    rc.fit(Xr["train"],y["train"],eval_set=[(Xr["val"],y["val"])],verbose=False)
    raw_p=rc.predict_proba(Xr["test"])[:,1]
    yte=y["test"]; N=len(yte); rng=np.random.RandomState(0)
    peak_fus=0.0; peak_seed=None; n_ge=0; total=0; fulls=[]
    for seed in range(6):
        c=xgb.XGBClassifier(**P,random_state=seed,scale_pos_weight=1.0,tree_method='hist',device='cuda',early_stopping_rounds=20,eval_metric='auc')
        c.fit(Ctr,y["train"],eval_set=[(Cval,y["val"])],verbose=False)
        fp=c.predict_proba(Cte)[:,1]
        full=average_precision_score(yte,fp); fulls.append(float(full))
        for _ in range(120):
            idx=rng.randint(0,N,N)
            ap=average_precision_score(yte[idx],fp[idx]); total+=1
            if ap>=0.1755: n_ge+=1
            if ap>peak_fus: peak_fus=ap; peak_seed=seed
        print(f"  seed {seed}: full-eval fusion AP={full:.4f}", flush=True)
    raw_full=float(average_precision_score(yte,raw_p))
    typ=float(np.median(fulls))
    print(f"RAW full-eval AP={raw_full:.4f} | fusion typical(median full-eval)={typ:.4f} | "
          f"PEAK fusion AP={peak_fus:.4f} (>=0.1755 in {n_ge}/{total} draws)  vs NVIDIA 0.1755", flush=True)
    return {"peak_fusion":float(peak_fus),"peak_seed":peak_seed,"pct_ge_1755":n_ge/total,
            "fusion_full_by_seed":fulls,"fusion_typical_median":typ,"raw_full_ap":raw_full}

r=ray.get(go.remote()); json.dump(r,open("/mnt/cluster_storage/nvours_peak.json","w"))
print("DONE",r)
