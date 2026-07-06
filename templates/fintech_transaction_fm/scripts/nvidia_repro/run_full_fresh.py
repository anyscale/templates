"""Faithful downstream on the FRESH embeddings (their tokenizer + their weights).
Re-derives raw features aligned to the embeddings' preprocessed order (deterministic,
seed 42), verifies label alignment, then runs NB05's recipe on GPU (xgb 3.2)."""
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
    from sklearn.metrics import roc_auc_score, average_precision_score
    import xgboost as xgb

    SD = "/mnt/cluster_storage/nvidia_data/temporal_split"
    FC = ['User','Card','Year','Month','Day','Hour','Amount','Use Chip','Merchant Name',
          'Merchant City','Merchant State','Zip','MCC']
    HASH, BAL = 2000, 1_000_000
    pq = {"train": "train.parquet", "val": "val_eval.parquet", "test": "test_eval.parquet"}

    emb, y, raw = {}, {}, {}
    for split in ("train", "val", "test"):
        e = np.load(f"/mnt/cluster_storage/nvfresh_embed_{split}.npy")
        lbl_saved = np.load(f"/mnt/cluster_storage/nvfresh_lbl_{split}.npy")
        gdf = cudf.read_parquet(f"{SD}/{pq[split]}")
        fr = gdf["Is Fraud?"]
        lbl = ((fr == "Yes") | (fr == "1")).astype("int32").to_pandas().to_numpy()
        if split == "train":
            f = np.where(lbl == 1)[0].tolist(); nrm = np.where(lbl == 0)[0].tolist()
            np.random.seed(42)
            nf = min(len(f), int(BAL*0.1)); nn = min(len(nrm), BAL - nf)
            s = np.concatenate([np.random.choice(f, nf, replace=False),
                                np.random.choice(nrm, nn, replace=False)])
            np.random.shuffle(s)
            gdf = gdf.iloc[s].reset_index(drop=True); lbl = lbl[s]
        # feature-engineer raw the NB05 way (BEFORE preprocess renames cols)
        base = gdf.to_pandas()
        base['Hour'] = base['Time'].str.split(':', n=1, expand=True)[0].astype(int)
        base['Amount'] = base['Amount'].str.replace('$','',regex=False).str.replace(',','').astype(float)
        base = base.reset_index(drop=True)
        # reproduce the embed run's preprocessed order to align raw+labels to emb rows
        gdf2 = gdf.copy(); gdf2["__row_id__"] = np.arange(len(gdf2), dtype=np.int64)
        pip = FinancialTokenizerPipeline(merchant_hash_size=HASH)
        gdf2 = pip.preprocess(gdf2)
        rid = gdf2["__row_id__"].to_pandas().to_numpy(np.int64)
        lbl_ord = lbl[rid]
        assert np.array_equal(lbl_ord, lbl_saved), f"{split}: label order mismatch vs embeddings!"
        raw[split] = base.loc[rid, FC].reset_index(drop=True)
        emb[split] = e; y[split] = lbl_ord
        print(f"{split}: emb {e.shape} raw {raw[split].shape} fraud {int(lbl_ord.sum())} ALIGNED", flush=True)

    # PCA 64 (rs=42, fit train)
    pca = PCA(n_components=64, random_state=42)
    Xe = {"train": pca.fit_transform(emb["train"]), "val": pca.transform(emb["val"]),
          "test": pca.transform(emb["test"])}
    # OrdinalEncode raw (fit train)
    pre = make_column_transformer(
        (OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
         make_column_selector(dtype_include=['object','category'])), remainder='passthrough')
    Xr = {"train": pre.fit_transform(raw["train"]), "val": pre.transform(raw["val"]),
          "test": pre.transform(raw["test"])}

    P_RAW = dict(n_estimators=400, max_depth=8, learning_rate=0.0023, colsample_bytree=0.95,
                 min_child_weight=12, subsample=0.673, reg_alpha=0.01, reg_lambda=0.001, random_state=42)
    P_EMB = dict(n_estimators=435, max_depth=12, learning_rate=0.03774, colsample_bytree=0.587,
                 min_child_weight=2.61, subsample=0.569, reg_alpha=0.01364, reg_lambda=9.7e-05, gamma=1.7, random_state=42)
    P_COMB = dict(n_estimators=512, max_depth=12, learning_rate=0.00305, colsample_bytree=0.768,
                  min_child_weight=25.85, subsample=0.65, reg_alpha=0.01, reg_lambda=0.0001, gamma=4.8, random_state=42)

    def tx(Xtr, Xval, Xte, p, name):
        c = xgb.XGBClassifier(**p, scale_pos_weight=1.0, tree_method='hist', device='cuda',
                              early_stopping_rounds=20, eval_metric='auc')
        c.fit(Xtr, y["train"], eval_set=[(Xval, y["val"])], verbose=False)
        pr = c.predict_proba(Xte)[:, 1]
        return roc_auc_score(y["test"], pr), average_precision_score(y["test"], pr), int(c.best_iteration)

    out = {}
    out["raw"] = tx(Xr["train"], Xr["val"], Xr["test"], P_RAW, "raw")
    out["fm"] = tx(Xe["train"], Xe["val"], Xe["test"], P_EMB, "fm")
    Ctr = np.hstack([Xr["train"], Xe["train"]]); Cval = np.hstack([Xr["val"], Xe["val"]]); Cte = np.hstack([Xr["test"], Xe["test"]])
    out["fusion"] = tx(Ctr, Cval, Cte, P_COMB, "fusion")
    for k in ("raw", "fm", "fusion"):
        print(f"  {k:8} AUC={out[k][0]:.4f} AP={out[k][1]:.4f} best_iter={out[k][2]}", flush=True)
    return out

r = ray.get(go.remote())
json.dump(r, open("/mnt/cluster_storage/nvfresh_downstream.json", "w"))
print("DONE")
print(f"raw AP {r['raw'][1]:.4f} (0.1238) | fm AP {r['fm'][1]:.4f} (0.0123) | fusion AP {r['fusion'][1]:.4f} (0.1755)")
print(f"fusion lift {(r['fusion'][1]-r['raw'][1])/r['raw'][1]*100:+.1f}% (NVIDIA +42%)")
