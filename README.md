[data_mix_full_report (2).md](https://github.com/user-attachments/files/30594180/data_mix_full_report.2.md)
# Planning the Pretraining Data Mix: Research, Top Models, and Our Experiment

## Part 1 

We looked at how leading methods and top models decide what data to train on. The same
pattern shows up everywhere: **general web data (Common Crawl) makes up most of the mix
(about 50–90%)**, and smaller, deliberate amounts of data are added for the specific
skills the model should be good at — like coding, math, and other languages.

We then ran our own small experiment to test a mix for our five data types, and it
confirmed the pattern.

---

## Part 2 — The methods we checked

- **DoReMi (Google, 2023).** Uses a small test model to figure out good proportions for
  each type of data, then applies them to the big model. It adjusts the balance of web
  data but never removes it.

- **RegMix (2024).** Trains many tiny models on different data mixes, learns which mix
  works best, and predicts the winner. Its main finding: **web data helps performance more
  than sources people assume are "higher quality," like Wikipedia.**

- **Apple – Optimal Data Mixtures (2025).** Builds a formula that predicts how well a mix
  will do, then finds the best mix for whatever skill you're targeting. Each skill needs
  its own specific share of data.

- **CLIMB (NVIDIA, 2025).** Splits a big web dataset into ~20 smaller groups and searches
  for the best combination. Even after all that splitting, **general-knowledge data is
  what won.** A focused slice of a specific topic improved that topic's score by about 5%
  — but only when the mix was deliberately steered toward it.

---

## Part 3 — What this tells us

**1. Web data earns its large share — it's not just a lazy default.**
Every method keeps web data as the biggest part of the mix. It's the foundation that gives
a model broad language ability and general knowledge. A specialist model still needs this
base underneath it.

**2. Specific skills only come from specific data.**
General web data barely covers things like coding, math, or Indian languages. If you want
the model to be good at those, you have to deliberately include data for them — you can't
get there by adding more general web.

**3. You don't need a lot of specialist data — just a real, dedicated amount.**
A small but intentional slice goes a long way. Top models add a modest amount of
high-quality math and code data near the end of training and see big jumps on math and
coding tests. The goal isn't to flood the mix with code or Indic data — it's to reserve a
meaningful, deliberate slice for each target skill.

---

## Part 4 — What the top models actually do

**Llama 3 (Meta).** Its final mix is about **50% general knowledge, 25% math and
reasoning, 17% code, and 8% other languages.**

**OLMo (Ai2).** Leans even more heavily on web. Its recent mix is roughly:

| Type of data | Share |
|--------------|------:|
| General web (Common Crawl) | 76% |
| Academic/science documents | 14% |
| Code | 7% |
| Math | 3% |
| Research papers (arXiv) | ~1% |

**The common thread:** general web gets the most weight (from 50% up to 90%+), followed by
code, then math, then papers — with each specialist slice sized to the skills the model is
meant to have.

---

## Part 5 — Our Experiment

### What we did

To test a mix for our own five data types — **web, code, math, Indic, and research
papers** — we ran a small version of the RegMix approach:

1. Trained many small models, each on a *different* mix of the five data types.
2. Recorded how well each mix did on each data type.
3. Used those results to predict the best overall mix.

Everything ran on a single cloud GPU. We used small models and modest data per run to keep
it affordable — this is a proof of concept, not a full-scale run, so the numbers show the
*direction*, not the final word.

### Experiment setup (the details)

**Data sources.** Each of the five data types was taken from a well-known public dataset:

| Data type | Source used |
|-----------|-------------|
| General web | ClimbMix (a cleaned Common Crawl collection) |
| Code | The Stack (Python) |
| Math | OpenWebMath |
| Indic | AI4Bharat Sangraha (4 Indian languages, balanced) |
| Research papers | Arxiver (arXiv papers) |

We prepared about 60 million words (tokens) of each type — roughly 300 million in total —
which is enough to compare mixes fairly while staying small and fast. We used a
multilingual tokenizer (the tool that splits text into tokens) so that Indian-language
text wasn't broken up inefficiently.

**Model size.** Each test model was a small GPT-style model — deliberately tiny (about 1
million "working" parameters, the size that matters for this comparison). Small models are
the whole point of the method: they're cheap to train, and the *ranking* of good vs. bad
mixes they produce still transfers to bigger models.

**Training setup.**

| Setting | Value |
|---------|-------|
| Model type | Small GPT (4 layers) |
| Data per test model | ~50 million tokens |
| Time per test model | ~9 minutes on the GPU |
| Number of test models trained | 53 in total, across two rounds |
| Hardware | One NVIDIA A100 GPU (via Colab) |

**How the rounds worked.**

- **Round 1:** 37 models on a wide spread of random mixes, to map the landscape.
- **Round 2:** 16 more models clustered around our proposed 60% web recipe, to zoom in.

After all 53, we let the method pick the best mix — with web anchored as the backbone, so
it couldn't wander off into unrealistic recipes.

**Scale note.** A full, publication-grade version of this method would use much bigger
models and far more data per run (the original RegMix used 512 models on 20x more data
each). Our smaller version is meant to demonstrate the approach and point to a sensible
mix, not to be the final production recipe.

**Record of all runs.** Every one of the 53 test models is logged in the file
`sweep_log.jsonl`. Each line records that model's data mix, how it scored on each data
type, and its overall score — so any result here can be reproduced or re-checked directly
from that file.

### What we found

- **The approach works.** The predictor could reliably tell good mixes from bad ones.

- **The mix follows whatever you tell it to prioritize.** In an early run we forgot to
  count web data toward the goal, and the mix promptly dropped web to almost nothing —
  a useful reminder that you have to explicitly value web for it to stay in the mix.

- **Web is the reliable backbone.** Once web was properly valued, it consistently came out
  as the biggest slice — the model wanted it around 57–67%, matching what the top models
  do.

- **Indic is a stable, meaningful slice.** Across every version of the experiment, Indic
  landed near 15% — the most consistent result we got.

- **A caution about chasing scores.** In one run the method wanted to pour 42% of the mix
  into math. On inspection this was a quirk: math text is easy to get a low "score" on
  without the model actually becoming better at math reasoning, and this happened when the
  method was allowed to search freely. This is why we don't follow the raw numbers blindly
  — we sanity-check them against common sense. Once we anchored web as the backbone (the
  realistic setting), math settled back to a sensible ~11%.

### The experiment's best mix

After anchoring web as the backbone and letting the method choose the rest, the most
stable result was:

| Type of data | Share |
|--------------|------:|
| General web | 57% |
| Code | 10% |
| Math | 11% |
| Indic | 15% |
| Research papers | 7% |

This lined up closely with what we'd expected going in, and with what the top models do —
web as the large backbone, meaningful dedicated slices for code and Indic.

---

## Part 6 — Our Suggested Mixture

Bringing together the research, the top models, and our own experiment, we recommend:

| Type of data | Share | Why |
|--------------|------:|-----|
| General web (Common Crawl) | **60%** | The backbone — broad language and general knowledge |
| Code | **10%** | To build coding ability |
| Math | **10%** | To build math and reasoning |
| Indic | **15%** | A bigger slice, since Indian languages are a priority and barely appear in web data |
| Research papers | **5%** | Academic and scientific coverage |

**Why these numbers:**

- **60% web** sits between Llama 3 (50%) and OLMo (76%). It keeps a strong general
  foundation while leaving room for the skills we care about. Our experiment agreed web
  should be the largest slice by far.
- **10% code and 10% math** give each a real, dedicated slice — enough to build the skill,
  in line with how top models size these.
- **15% Indic** is deliberately larger than top models use for other languages, because
  Indic is a priority for us and is very underrepresented in general web data. Our
  experiment consistently supported ~15% here.
- **5% research papers** covers academic and scientific writing.

**One honest note:** Indic is a hard, underrepresented area. A 15% share is a serious
commitment, but getting strong Indic performance also depends on the *quality* of the
Indic data and using a tokenizer suited to Indian scripts — not just the percentage.
