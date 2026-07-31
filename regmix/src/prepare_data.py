import os, json, sys, glob, numpy as np
from itertools import islice
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

ROOT = "/content/regmix"
cfg  = json.load(open(f"{ROOT}/src/config.json"))
BIN  = f"{ROOT}/data_bin"; os.makedirs(BIN, exist_ok=True)

tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
DTYPE = np.uint16 if tok.vocab_size < 65536 else np.uint32
print(f"tokenizer={cfg['tokenizer']} vocab={tok.vocab_size} dtype={DTYPE.__name__}")

def write_bin(path, token_iter, max_tokens):
    """Stream tokens into a growable memmap; stop at max_tokens."""
    buf = np.memmap(path, dtype=DTYPE, mode="w+", shape=(max_tokens,))
    n = 0
    pbar = tqdm(total=max_tokens, unit="tok", unit_scale=True, desc=os.path.basename(path))
    for ids in token_iter:
        if n >= max_tokens: break
        take = min(len(ids), max_tokens - n)
        buf[n:n+take] = ids[:take]; n += take; pbar.update(take)
    pbar.close(); buf.flush()
    # trim file to actual length
    actual = np.memmap(path, dtype=DTYPE, mode="r", shape=(n,))
    np.array(actual).tofile(path)
    print(f"  wrote {n:,} tokens -> {path}")
    return n

def hf_token_stream(src):
    ds = load_dataset(src["id"], src["config"], split=src["split"], streaming=True)
    for ex in ds:
        txt = ex.get(src["text_key"])
        if txt:
            yield tok.encode(txt) + [tok.eos_token_id or 0]

def local_token_stream(folder):
    files = sorted(glob.glob(f"{folder}/**/*.txt", recursive=True)) + \
            sorted(glob.glob(f"{folder}/**/*.jsonl", recursive=True))
    assert files, f"No .txt/.jsonl found in {folder} — upload your Indic data there first."
    for fp in files:
        if fp.endswith(".jsonl"):
            for line in open(fp, encoding="utf-8"):
                try: txt = json.loads(line).get("text","")
                except: txt = ""
                if txt: yield tok.encode(txt) + [tok.eos_token_id or 0]
        else:
            txt = open(fp, encoding="utf-8").read()
            for para in txt.split("\n\n"):
                if para.strip(): yield tok.encode(para) + [tok.eos_token_id or 0]


def indic_sangraha_stream():
    ic = cfg["indic_sangraha"]
    import itertools
    gens = []
    for lang in ic["langs"]:
        ds = load_dataset(ic["id"], data_dir=f'{ic["subset"]}/{lang}',
                          split="train", streaming=True)
        gens.append((lang, iter(ds), ic["tokens_per_lang"]))
    # round-robin across languages, each capped at tokens_per_lang
    counts = {l: 0 for l, _, _ in gens}
    active = list(gens)
    while active:
        nxt = []
        for lang, g, cap in active:
            if counts[lang] >= cap:
                continue
            try:
                ex = next(g)
            except StopIteration:
                continue
            txt = ex.get(ic["text_key"])
            if txt:
                ids = tok.encode(txt) + [tok.eos_token_id or 0]
                counts[lang] += len(ids)
                yield ids
            nxt.append((lang, g, cap))
        active = nxt
    print("  indic per-language tokens:", counts)

def prepare_domain(name):
    train_tok = cfg["tokens_per_domain"]
    val_tok   = cfg["val_tokens_per_domain"]
    total     = train_tok + val_tok
    if name == "indic":
        stream = indic_sangraha_stream()
    else:
        stream = hf_token_stream(cfg["hf_sources"][name])
    # write combined then split
    tmp = f"{BIN}/{name}_all.bin"
    n = write_bin(tmp, stream, total)
    arr = np.fromfile(tmp, dtype=DTYPE)
    n_val = min(val_tok, n // 10)
    arr[n_val:].tofile(f"{BIN}/{name}_train.bin")
    arr[:n_val].tofile(f"{BIN}/{name}_val.bin")
    os.remove(tmp)
    return {"name": name, "train": int(n - n_val), "val": int(n_val)}

if __name__ == "__main__":
    which = sys.argv[1:] or cfg["domains"]
    manifest = {}
    for name in which:
        print(f"\n=== {name} ===")
        manifest[name] = prepare_domain(name)
    # merge into any existing manifest
    mpath = f"{BIN}/manifest.json"
    old = json.load(open(mpath)) if os.path.exists(mpath) else {}
    old.update(manifest); json.dump(old, open(mpath,"w"), indent=2)
    print("\nManifest:", json.dumps(old, indent=2))
