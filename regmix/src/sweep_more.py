import os, json, sys, time, numpy as np
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
sys.path.insert(0,"/content/regmix/src")
import model as M
ROOT="/content/regmix"; cfg=json.load(open(f"{ROOT}/src/config.json"))
DOMAINS=cfg["domains"]; LOG=f"{ROOT}/runs/sweep_log.jsonl"; TW=cfg["target_weights"]

N_NEW   = 20
ALPHA   = 1.0        # broad Dirichlet, same as original sweep
START_ID = 2000      # distinct from sweep (0-36) and refine (1000-1015)

def target_metric(vl): return sum(TW[d]*vl[d] for d in DOMAINS)
def done_ids():
    if not os.path.exists(LOG): return set()
    return {json.loads(l)["id"] for l in open(LOG)}

rng=np.random.default_rng(500)                       # new seed
new_mix = rng.dirichlet([ALPHA]*len(DOMAINS), size=N_NEW)
done = done_ids()
print(f"{N_NEW} broad proxies (alpha={ALPHA}), ids {START_ID}-{START_ID+N_NEW-1}")
for j in range(N_NEW):
    jid = START_ID + j
    if jid in done:
        print(f"  skip {jid} (done)"); continue
    m = new_mix[j]
    t0=time.time()
    vl = M.train_one_proxy(m.tolist(), log_every=10**9)
    rec={"id":jid,"mixture":m.tolist(),"vloss":vl,
         "target":target_metric(vl),"sec":round(time.time()-t0,1),"phase":"sweep2"}
    with open(LOG,"a") as f: f.write(json.dumps(rec)+"\n")
    print(f"[sweep2 {j+1}/{N_NEW}] id={jid} target={rec['target']:.4f} "
          f"mix={[round(x,3) for x in m.tolist()]} ({rec['sec']}s)", flush=True)
print("done.")
