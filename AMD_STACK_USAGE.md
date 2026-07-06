# 🚀 AMD Stack Usage + Auto-Deploy Setup

---

## 1. How Our Code Uses Each AMD Technology

### 🔷 AMD Developer Cloud (MI300X GPU, $100 credits)

**What it is:** On-demand access to AMD GPUs for training, fine-tuning, inference.

**How Kernel Olympics uses it:**

```
┌─────────────────────────────────────────────┐
│  AMD Developer Cloud (MI300X GPU)            │
│                                              │
│  Our Verification Agent (src/verification/)  │
│  runs on this GPU:                           │
│                                              │
│  1. Pull Docker container with ROCm stack    │
│  2. hipcc compiles ported HIP kernel         │
│  3. Kernel executes on MI300X                │
│  4. Output captured and diffed against       │
│     CUDA reference                           │
│                                              │
│  This is the "proof of correctness" step —   │
│  without real AMD hardware, we can't prove   │
│  the ported kernel actually works.           │
└─────────────────────────────────────────────┘
```

**Code entry point:** `src/verification/verifier.py` — `VerificationAgent.verify()`
- Calls `hipcc` to compile
- Runs binary on AMD GPU
- Diffs output byte-for-byte

**Cost:** ~$5/hour on MI300X. We need ~20 hours total = $100 credits.

---

### 🔥 Fireworks AI API ($50 credits)

**What it is:** API access to AMD-hardware hosted models (Kimi, GLM, Gemma).

**How Kernel Olympics uses it:**

```
┌─────────────────────────────────────────────┐
│  Fireworks AI API ($50 credits)              │
│                                              │
│  Our Porting Agent (src/porting_agent/)      │
│  calls Fireworks for the SMART part:         │
│                                              │
│  Kimi K2.7 → Analyzes CUDA kernel structure  │
│              (architecture, memory pattern,   │
│               warp usage, library calls)      │
│                                              │
│  GLM-5.4   → Generates ROCm/HIP equivalent   │
│              code with annotations per change │
│                                              │
│  This is only called for RED-flagged kernels  │
│  (the hard 20% that hipify can't handle).     │
│  Green/yellow kernels use template fallback   │
│  with zero API cost.                          │
└─────────────────────────────────────────────┘
```

**Code entry point:** `src/porting_agent/agent.py` — `PortingAgent.port_kernel()`
- Calls `POST https://api.fireworks.ai/inference/v1/chat/completions`
- Uses ~500 tokens per kernel = ~$0.001/call
- 500 calls max = $0.50 (we have $50 — way under budget)

---

### 🟢 ROCm (AMD GPU Computing — open source)

**What it is:** AMD's open-source GPU platform. Runs PyTorch, TensorFlow, custom HIP kernels.

**How Kernel Olympics uses it:**

```
┌─────────────────────────────────────────────┐
│  ROCm                                       │
│                                              │
│  Used EVERYWHERE in our pipeline:            │
│                                              │
│  1. Docker base image:                       │
│     rocm/dev-ubuntu-22.04:latest             │
│     → Includes hipcc, rocm-dev, rocm-libs   │
│                                              │
│  2. hipcc compiles our ported HIP kernels    │
│     → --offload-arch=gfx942 (MI300X)        │
│                                              │
│  3. Gemma runs locally via ROCm:             │
│     → Report generator (plain-English        │
│       summary without API cost)              │
│                                              │
│  4. This is our GEMMA PRIZE angle:           │
│     "Best AMD-Hosted Gemma Project" = $2K    │
└─────────────────────────────────────────────┘
```

**Code entry point:**
- `Dockerfile` — `FROM rocm/dev-ubuntu-22.04:latest`
- `src/report_generator/reporter.py` — `ReportGenerator` with `use_gemma=True` flag

---

### 🤖 Gemma (Google DeepMind — lightweight open models)

**What it is:** Lightweight open models (2B, 7B) — Apache 2.0 license. Available via Fireworks API or local ROCm inference.

**How Kernel Olympics uses it:**

```
┌─────────────────────────────────────────────┐
│  Gemma (Google DeepMind)                     │
│                                              │
│  TWO use cases in our pipeline:              │
│                                              │
│  USE CASE 1 (main — Gemma Prize):            │
│  Gemma on LOCAL ROCm → Report Generator     │
│  Generates plain-English portability report: │
│  "Your kernel has 3 warp divergence risks.   │
│   We fixed the shuffle reduction. Estimated  │
│   4 engineer-hours saved."                   │
│  Zero API cost — runs on AMD GPU.            │
│                                              │
│  USE CASE 2 (fallback):                      │
│  Gemma via Fireworks API                     │
│  If local ROCm inference isn't ready,        │
│  we call Gemma through Fireworks credits.    │
│                                              │
│  Why this wins the Gemma Prize ($2K):        │
│  • Gemma runs ON AMD HARDWARE (ROCm)         │
│  • Gemma generates real value (not demo)     │
│  • Genuine use case: text generation for     │
│    technical report writing                  │
└─────────────────────────────────────────────┘
```

**Code entry point:** `src/report_generator/reporter.py`
- `_gemma_summary()` method — calls local Gemma via ROCm
- `_template_summary()` — fallback without Gemma

---

## 2. 💸 Budget Allocation

| Resource | Credits | Our Usage | Expected Spend |
|----------|:-------:|-----------|:--------------:|
| AMD Developer Cloud | $100 | ~20h GPU time for verification + Gemma inference | $100 |
| Fireworks AI API | $50 | ~500 LLM calls for porting (Kimi + GLM) | $0.50 |
| Gemma local ROCm | $0 | Runs on AMD GPU — zero API cost | $0 |
| **Total** | **$150** | | **~$100.50** |

---

## 3. 🚀 Auto-Deploy on AMD Cloud After GitHub Push

### Architecture

```
GitHub Push → GitHub Actions → SSH to AMD Cloud → Docker Build → Deploy
```

### Step-by-Step Setup

#### 3.1 — Create AMD Developer Cloud Instance

```bash
# 1. Sign up at https://www.amd.com/en/developer/ai-dev-program.html
# 2. Launch a GPU instance (MI300X recommended)
# 3. Note the SSH connection details:
#    Host: <your-instance-ip>
#    User: ubuntu (or amd-user)
#    Key: your-ssh-private-key
```

#### 3.2 — Install ROCm + Docker on AMD Instance

```bash
# SSH into AMD Cloud instance
ssh ubuntu@<instance-ip>

# Install ROCm
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install.deb
sudo apt install -y ./amdgpu-install.deb
sudo amdgpu-install --usecase=rocm

# Install Docker
sudo apt install -y docker.io
sudo usermod -aG docker $USER

# Verify ROCm
rocm-smi  # Should show MI300X GPU

# Logout and reconnect for group changes
exit
```

#### 3.3 — Set GitHub Secrets

In your repo → Settings → Secrets and variables → Actions:

| Secret | Value |
|--------|-------|
| `AMD_CLOUD_HOST` | `<your-instance-ip>` |
| `AMD_CLOUD_USER` | `ubuntu` |
| `AMD_CLOUD_SSH_KEY` | Private SSH key (PEM format) |
| `FIREWORKS_API_KEY` | Your Fireworks API key |
| `AMD_CLOUD_WORK_DIR` | `/home/ubuntu/kernel-olympics` |

#### 3.4 — Create GitHub Actions Workflow

```yaml
# .github/workflows/deploy-amd.yml
name: Deploy to AMD Cloud

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup SSH
        run: |
          mkdir -p ~/.ssh
          echo "${{ secrets.AMD_CLOUD_SSH_KEY }}" > ~/.ssh/id_rsa
          chmod 600 ~/.ssh/id_rsa
          ssh-keyscan -H ${{ secrets.AMD_CLOUD_HOST }} >> ~/.ssh/known_hosts

      - name: Copy code to AMD Cloud
        run: |
          rsync -avz --delete --exclude='.git' \
            ./ ${{ secrets.AMD_CLOUD_USER }}@${{ secrets.AMD_CLOUD_HOST }}:${{ secrets.AMD_CLOUD_WORK_DIR }}/

      - name: Build & test on AMD GPU
        run: |
          ssh ${{ secrets.AMD_CLOUD_USER }}@${{ secrets.AMD_CLOUD_HOST }} \
            "cd ${{ secrets.AMD_CLOUD_WORK_DIR }} && \
             docker build -t kernel-olympics . && \
             docker run --rm --device=/dev/kfd --device=/dev/dri \
               -v $PWD:/app \
               kernel-olympics \
               --input sample_kernels/cuda/warp_reduce.cu"

      - name: Run verification on real AMD GPU
        run: |
          ssh ${{ secrets.AMD_CLOUD_USER }}@${{ secrets.AMD_CLOUD_HOST }} \
            "cd ${{ secrets.AMD_CLOUD_WORK_DIR }} && \
             python3 -m src.main \
               --input sample_kernels/cuda/warp_reduce.cu \
               --output portability_report.json"
          
      - name: Fetch report back
        run: |
          scp ${{ secrets.AMD_CLOUD_USER }}@${{ secrets.AMD_CLOUD_HOST }}:${{ secrets.AMD_CLOUD_WORK_DIR }}/portability_report.json ./latest_report.json

      - name: Upload report as artifact
        uses: actions/upload-artifact@v4
        with:
          name: portability-report
          path: latest_report.json
```

#### 3.5 — Create the workflow file locally

```bash
mkdir -p .github/workflows
# Create deploy-amd.yml with the content above
git add .github/workflows/deploy-amd.yml
git commit -m "🚀 Auto-deploy to AMD Cloud on push"
git push
```

### Deploy Flow (Visual)

```
You push to main
       │
       ▼
GitHub Actions triggers
       │
       ├── rsync code → AMD Cloud instance
       │
       ├── Docker build (ROCm + Python)
       │
       ├── Run scanner + classifier on AMD GPU
       │
       ├── Run verification (real hipcc compile)
       │
       ├── Generate Gemma report
       │
       └── Fetch report → GitHub artifact
              │
              ▼
       You download from Actions tab
```

### What the Judge Sees (Live Demo)

```
[Terminal — Live on AMD Cloud]

$ python3 -m src.main --input sample_kernels/cuda/warp_reduce.cu

[1/6] Scanning warp_reduce.cu...               ← hipify running on AMD
[2/6] Classifying risk... RED — 8 patterns     ← rule-based (fast)
[3/6] Checking pattern memory... 0 stored      ← first run
[4/6] Porting warp_reduce.cu... confidence 85% ← Fireworks API
[5/6] Verifying on AMD MI300X...               ← REAL GPU COMPILE
          ✓ Compiled with hipcc
          ✓ Ran on AMD GPU
          ✓ Output matches CUDA reference
[6/6] Report generated. 4 engineer-hours saved.

Pattern memory: 1 pattern stored ✓
Demo: "This is running on real AMD silicon right now."
```

---

## Summary

| AMD Tech | How We Use It | File |
|----------|---------------|------|
| **AMD Cloud GPU** | Run ported kernels, verify correctness | `src/verification/verifier.py` |
| **Fireworks API** | Kimi + GLM for smart porting of hard kernels | `src/porting_agent/agent.py` |
| **ROCm** | GPU compute platform, hipcc compiler, Gemma inference | `Dockerfile`, `src/report_generator/reporter.py` |
| **Gemma** | Generate plain-English portability report (Gemma Prize!) | `src/report_generator/reporter.py` |
| **Auto-deploy** | GitHub Actions → rsync → Docker build → test on AMD Cloud | `.github/workflows/deploy-amd.yml` |
