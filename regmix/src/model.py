import os, json, math, numpy as np, torch, torch.nn as nn
from torch.nn import functional as F

ROOT = "/content/regmix"
cfg  = json.load(open(f"{ROOT}/src/config.json"))
BIN  = os.environ.get("REGMIX_BIN", f"{ROOT}/data_bin")
DOMAINS = cfg["domains"]
DTYPE = np.uint32
DEVICE = "cuda"

# ---------- model (tiny GPT, weight-tied) ----------
class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.h = h; s.d = d
        s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3*d)
        s.proj = nn.Linear(d, d)
        s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x, mask=None):
        B,T,D = x.shape
        q,k,v = s.qkv(s.ln1(x)).split(D, dim=2)
        q=q.view(B,T,s.h,D//s.h).transpose(1,2)
        k=k.view(B,T,s.h,D//s.h).transpose(1,2)
        v=v.view(B,T,s.h,D//s.h).transpose(1,2)
        a=F.scaled_dot_product_attention(q,k,v,is_causal=True)  # fused Flash kernel
        a=a.transpose(1,2).contiguous().view(B,T,D)
        x = x + s.proj(a)
        return x + s.mlp(s.ln2(x))

class TinyGPT(nn.Module):
    def __init__(s, vocab, d=128, h=4, L=4, block=512):
        super().__init__()
        s.block = block
        s.tok = nn.Embedding(vocab, d)
        s.pos = nn.Embedding(block, d)
        s.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        s.lnf = nn.LayerNorm(d)
        s.head = nn.Linear(d, vocab, bias=False)
        s.head.weight = s.tok.weight            # weight tying
        s.apply(s._init)
    def _init(s, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
    def forward(s, idx, targets=None):
        T = idx.size(1)
        x = s.tok(idx) + s.pos(torch.arange(T, device=idx.device))
        for b in s.blocks: x = b(x)
        h = s.lnf(x)
        if targets is None:
            return s.head(h), None
        h = h.view(-1, h.size(-1)); t = targets.view(-1)
        chunk = 4096; total = 0.0; n = h.size(0)
        for i in range(0, n, chunk):
            lg = s.head(h[i:i+chunk]).float()
            total = total + F.cross_entropy(lg, t[i:i+chunk], reduction="sum")
        return None, total / n

def count_non_embed(m):
    emb = m.tok.weight.numel() + m.pos.weight.numel()
    return sum(p.numel() for p in m.parameters()) - emb

# ---------- memmap loaders ----------
_mm = {}
def _get(name, split):
    k=(name,split)
    if k not in _mm:
        _mm[k]=np.memmap(f"{BIN}/{name}_{split}.bin", dtype=DTYPE, mode="r")
    return _mm[k]

def get_batch(mixture, split, bs, block):
    """Draw `bs` sequences; each sequence's domain is sampled from `mixture`."""
    counts = np.random.multinomial(bs, mixture)
    xs, ys = [], []
    for name, c in zip(DOMAINS, counts):
        if c==0: continue
        arr=_get(name, split); n=len(arr)
        starts=np.random.randint(0, n-block-1, size=c)
        for st in starts:
            chunk=arr[st:st+block+1].astype(np.int64)
            xs.append(chunk[:-1]); ys.append(chunk[1:])
    X=torch.from_numpy(np.stack(xs)).to(DEVICE)
    Y=torch.from_numpy(np.stack(ys)).to(DEVICE)
    return X, Y

# ---------- train one proxy from scratch ----------
def train_one_proxy(mixture, tokens=None, micro_bs=48, accum=2, lr=3e-3, log_every=200):
    torch.manual_seed(0); np.random.seed(0)
    p=cfg["proxy"]; block=p["block_size"]
    tokens = tokens or p["tokens_per_proxy"]
    vocab = int(max(_get(d,"train").max() for d in DOMAINS))+1
    model=TinyGPT(vocab, block=block).to(DEVICE)
    if os.environ.get("PRINT_PARAMS"):
        print(f"non-embed params: {count_non_embed(model):,} | total: {sum(x.numel() for x in model.parameters()):,}")
    opt=torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.95), weight_decay=0.1)
    eff_bs = micro_bs*accum
    steps = tokens // (eff_bs*block)
    model.train()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            X,Y=get_batch(mixture,"train",micro_bs,block)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _,loss=model(X,Y)
            (loss/accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step % log_every==0:
            print(f"  step {step}/{steps} loss {loss.item():.3f}", flush=True)
    # per-domain val loss (pure domain mixtures)
    model.eval(); vloss={}
    with torch.no_grad():
        for i,name in enumerate(DOMAINS):
            onehot=np.zeros(len(DOMAINS)); onehot[i]=1.0
            ls=[]
            for _ in range(20):
                X,Y=get_batch(onehot,"val",16,block)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _,l=model(X,Y)
                ls.append(l.item())
            vloss[name]=float(np.mean(ls))
    del model, opt
    torch.cuda.empty_cache()
    return vloss

if __name__=="__main__":
    os.environ["PRINT_PARAMS"]="1"
    print("Domains:", DOMAINS)
    vl=train_one_proxy([0.2]*5, tokens=2_000_000)   # tiny smoke test
    print("val losses:", json.dumps(vl, indent=2))
