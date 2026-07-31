import os, json, sys, time, numpy as np
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
sys.path.insert(0, "/content/regmix/src")
import model as M

ROOT="/content/regmix"; cfg=json.load(open(f"{ROOT}/src/config.json"))
DOMAINS=cfg["domains"]; RUNS=f"{ROOT}/runs"; os.makedirs(RUNS,exist_ok=True)
LOG=f"{RUNS}/sweep_log.jsonl"
TW=cfg["target_weights"]

def target_metric(vloss):
    return sum(TW[d]*vloss[d] for d in DOMAINS)

def sample_mixtures(n, alpha, seed):
    rng=np.random.default_rng(seed)
    mix=rng.dirichlet([alpha]*len(DOMAINS), size=n)
    # add a few near-extreme mixtures so the regressor sees the edges
    for i in range(len(DOMAINS)):
        e=np.full(len(DOMAINS),0.02); e[i]=1-0.02*(len(DOMAINS)-1); mix=np.vstack([mix,e])
    return mix

def done_ids():
    if not os.path.exists(LOG): return set()
    return {json.loads(l)["id"] for l in open(LOG)}

def run_sweep():
    s=cfg["sweep"]
    mix=sample_mixtures(s["n_mixtures"], s["dirichlet_alpha"], s["seed"])
    done=done_ids()
    print(f"{len(mix)} mixtures total, {len(done)} already done")
    for i,m in enumerate(mix):
        if i in done:
            continue
        t0=time.time()
        vloss=M.train_one_proxy(m.tolist(), log_every=999999)  # quiet per-proxy
        rec={"id":i,"mixture":m.tolist(),"vloss":vloss,
             "target":target_metric(vloss),"sec":round(time.time()-t0,1)}
        with open(LOG,"a") as f: f.write(json.dumps(rec)+"\n")   # atomic append
        print(f"[{i+1}/{len(mix)}] target={rec['target']:.3f} ({rec['sec']}s)", flush=True)
    print("sweep complete.")

def fit_and_search():
    import lightgbm as lgb
    from scipy.stats import spearmanr
    recs=[json.loads(l) for l in open(LOG)]
    X=np.array([r["mixture"] for r in recs]); y=np.array([r["target"] for r in recs])
    # holdout to sanity-check the predictor
    n=len(X); idx=np.random.default_rng(0).permutation(n)
    cut=int(n*0.8); tr,te=idx[:cut],idx[cut:]
    dtrain=lgb.Dataset(X[tr],y[tr])
    params=dict(objective="regression",num_leaves=15,min_data_in_leaf=5,
                learning_rate=0.05,lambda_l1=0.1,lambda_l2=0.1,verbose=-1)
    gbm=lgb.train(params,dtrain,num_boost_round=500)
    pred=gbm.predict(X[te])
    rho=spearmanr(pred,y[te]).correlation
    print(f"held-out rank corr (Spearman): {rho:.3f}  (want > 0.7)")
    # refit on all data, search 100k candidates
    gbm=lgb.train(params,lgb.Dataset(X,y),num_boost_round=500)
    cand=np.random.default_rng(1).dirichlet([1.0]*len(DOMAINS),size=100_000)
    p=gbm.predict(cand)
    best=cand[p.argmin()]
    print("\nOPTIMAL MIXTURE (minimizes weighted target):")
    for d,w in zip(DOMAINS,best): print(f"  {d:8s} {w:.3f}")
    json.dump({"domains":DOMAINS,"optimal":best.tolist()},
              open(f"{RUNS}/optimal_mixture.json","w"),indent=2)
    return best

if __name__=="__main__":
    if sys.argv[1:]==["search"]:
        fit_and_search()
    else:
        run_sweep()
        fit_and_search()
