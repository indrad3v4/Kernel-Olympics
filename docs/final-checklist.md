# Final Checklist — Before Submission

## 🔴 MUST DO NOW (Order matters)

### 1. Submit on lablab.ai
⏱️ 10 min — DO THIS FIRST

### 2. State in submission:
- "Verified via real compilation + execution on AMD MI300X"
- "Gemma Prize ($6,000) eligible — Gemma 4 attempted on AMD GPU via vLLM (fallback: DeepSeek verified)"
- "4-model MOA: DeepSeek v4 Pro(planner) → GLM-5.2(coder) → Kimi K2.7(evaluator) + Gemma 4/DeepSeek(verifier fallback)"

### 3. Add "vs static analysis" section
Their verify = syntax check. Our verify = real hipcc compile + MI300X run + numeric diff.
Say it explicitly: "We don't guess. We compile and run."

## 🟡 HIGH (if time)

### 4. Test against NVIDIA/cuda-samples
Pick ONE file from NVIDIA/cuda-samples repo (e.g. warp_reduce or simple shuffle kernel)
Run through pipeline → capture output
Add to docs/nvidia-cuda-sample-proof.md

### 5. Simple web page (30 min)
Single HTML file wrapping CLI output with ASD color scheme
No build step, no server — just open index.html
Reuse the dashboard HTML you already have

### 6. Record demo video (10 min)
QuickTime: rocm-smi → pipeline → hipcc → PASSED
Upload to lablab

## 🟢 NICE (if double time)

### 7. CI green (wait for webhook)

### 8. Add "AMD Compatible" badge to README ✅ (already done)

---

## Summary for Discord (@team)

@team judge feedback received. Priorities:
1. Submit on lablab NOW — nothing else matters
2. In submission: state "real AMD GPU compilation + execution, not static analysis"
3. In submission: state "Gemma Prize eligible (attempting AMD GPU verifier; DeepSeek fallback deployed)"
4. If time: test against 1 file from NVIDIA/cuda-samples
5. If double time: wrap dashboard in simple HTML page

I'll write the submission text. Who's submitting?
