# ECG Heartbeat CBM & CEM Explorer — Vercel Static Deployment

Zero-cold-start, static CDN build of the interactive **ECG Heartbeat CBM & CEM Explorer** supporting both **PTB-XL Fold 10** and **Georgia 12-Lead ECG (Zero-Shot OOD)** cohorts (`30` curated clinical recordings across `NORM`, `AFIB`, `LBBB/RBBB`, `MI`, `STTC`, `HYP`, `IAVB`, `LQT`, and all 4 Georgia quality tiers).

## Why Static + Exact Linear Head Export?

PyTorch CPU wheels exceed Vercel's `250 MB` serverless limit and trigger slow cold starts. Instead, [explorer/build_vercel.py](../build_vercel.py) exports:
1. Precomputed `500 Hz` sliding-window concept probabilities (`stride = 25` samples = `50 ms`), batched **MC-Dropout Epistemic** decompositions, exact per-window **CEM** positive/negative logits `(a⁺, a⁻)`, and **CBM** linear classifier weights `(w_prob, w_var, b)` so **all test-time window concept interventions execute deterministically in browser JavaScript with `< 1e-4` error relative to PyTorch**.
2. Precomputed `100 Hz` **DenseCEM / DenseCBM / DenseMultiTask / DenseBlackBox** `uint8`-quantized soft segmentations, precomputed wave-shading spans, MC-Dropout boundary uncertainty `σ(t)`, clinical `PR / QRS / QT / HR` intervals, and both full-record and `1.0 s` window-scoped wave-mask interventions (`ablate` / `force` / `Apply GT Masks`).

You can also connect a live FastAPI GPU/CPU backend at any time by appending `?api=https://your-fastapi-host` to the deployed Vercel URL.

## Deploy to Vercel

### Option 1 — Vercel Dashboard (Recommended)
1. Push this repository to GitHub/GitLab/Bitbucket.
2. Import the repository in the Vercel Dashboard.
3. Set **Root Directory** to `explorer/vercel`, leave **Framework Preset** as `Other`, leave **Build Command** empty, and click **Deploy**.

### Option 2 — Vercel CLI
```bash
cd explorer/vercel
npx vercel --prod
```

## Regenerating the Static Bundle

Whenever checkpoints, showcase records, or benchmark summaries change, regenerate `explorer/vercel/`:

```bash
CUDA_VISIBLE_DEVICES=0 .venv_new/bin/python explorer/build_vercel.py
```

And preview locally before deploying:

```bash
python3 -m http.server 8041 --directory explorer/vercel
```
