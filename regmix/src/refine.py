import os, json, sys, time, numpy as np
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
sys.path.insert(0,"/content/regmix/src")
import model as M
ROOT="/content/regmix"; cfg=json.load(open(f"{ROOT}/src/config.json"))
DOMAINS=cfg["domains"]; LOG=f"{ROOT}/runs/sweep_log.jsonl"; TW=cfg["target_weights"]

CENTER = {"web":0.60,"code":0.10,"maths":0.10,"indic":0.15,"papers":0.05}
center = np.array([CENTER[d] for d in DOMAINS])
N_NEW  = 16
CONC   = 200.0

def target_metric(vl): return sum(TW[d]*vl[d] for d in DOMAINS)
def done_ids():
    if not os.path.exists(LOG): return set()
    return {json.loads(l)["id"] for l in open(LOG)}

rng=np.random.default_rng(100)
new_mix = rng.dirichlet(center*CONC, size=N_NEW)
done = done_ids()
start_id = 1000
print(f"{N_NEW} refinement proxies around {CENTER}, conc={CONC}")
for j in range(N_NEW):
    jid = start_id + j
    if jid in done:
        print(f"  skip {jid} (done)"); continue
    m = new_mix[j]
    t0=time.time()
    vl = M.train_one_proxy(m.tolist(), log_every=10**9)
    rec={"id":jid,"mixture":m.tolist(),"vloss":vl,
         "target":target_metric(vl),"sec":round(time.time()-t0,1),"phase":"refine"}
    with open(LOG,"a") as f: f.write(json.dumps(rec)+"\n")
    print(f"[refine {j+1}/{N_NEW}] id={jid} target={rec['target']:.4f} "
          f"mix={[round(x,3) for x in m.tolist()]} ({rec['sec']}s)", flush=True)
print("refinement complete.")
