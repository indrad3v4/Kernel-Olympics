const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Team Meteorite";
pres.title = "Kernel Olympics — CUDA to ROCm Migration Copilot";

// ── AMD Color Palette ──
const C = {
  bg:      "1A1A2E",  // deep navy bg
  card:    "16213E",  // slightly lighter navy
  accent:  "ED1C24",  // AMD red
  accent2: "E94560",  // warm accent
  text:    "FFFFFF",  // white text
  muted:   "8899AA",  // muted gray-blue
  cyan:    "0F3460",  // dark cyan
  highlight: "FF6B6B", // soft red highlight
  green:   "4ECB71",
  amber:   "FFB347",
};

// ── Helpers ──
function makeShadow() {
  return { type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.3 };
}

function addFooter(slide, text) {
  slide.addText(text || "Kernel Olympics — Team Meteorite 🌠", {
    x: 0.5, y: 5.0, w: 9, h: 0.4,
    fontSize: 9, color: C.muted, fontFace: "Calibri", align: "center",
  });
}

function addPageNum(slide, num, total) {
  slide.addText(`${num} / ${total}`, {
    x: 8.5, y: 5.0, w: 1.2, h: 0.35,
    fontSize: 9, color: C.muted, fontFace: "Calibri", align: "right",
  });
}

function addCard(slide, x, y, w, h, fillColor) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: fillColor || C.card },
    shadow: makeShadow(),
  });
}

function addTitleBar(slide, text) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.08,
    fill: { color: C.accent },
  });
  slide.addText(text, {
    x: 0.5, y: 0.2, w: 9, h: 0.6,
    fontSize: 22, fontFace: "Calibri", color: C.text, bold: true,
    margin: 0,
  });
}

// ── SLIDE 1: Title ──
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };

  // Large accent bar left
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.12, h: 5.625, fill: { color: C.accent },
  });

  // Title
  s.addText("KERNEL OLYMPICS", {
    x: 0.7, y: 1.0, w: 8.5, h: 1.2,
    fontSize: 44, fontFace: "Calibri", color: C.text, bold: true,
    charSpacing: 4, margin: 0,
  });

  // Subtitle
  s.addText("CUDA → ROCm Migration Copilot", {
    x: 0.7, y: 2.2, w: 8.5, h: 0.7,
    fontSize: 24, fontFace: "Calibri", color: C.accent, bold: false,
    margin: 0,
  });

  // Tagline
  s.addText("Ship AMD-ready code in minutes, not months.", {
    x: 0.7, y: 2.9, w: 8.5, h: 0.5,
    fontSize: 16, fontFace: "Calibri Light", color: C.muted,
    margin: 0,
  });

  // Divider line
  s.addShape(pres.shapes.LINE, {
    x: 0.7, y: 3.5, w: 3.0, h: 0,
    line: { color: C.accent, width: 2 },
  });

  // Event + team
  s.addText([
    { text: "AMD Developer Hackathon ACT II  •  Track 3 — Unicorn", options: { breakLine: true, color: C.muted, fontSize: 13 } },
    { text: "Team Meteorite 🌠  •  Team-3793", options: { color: C.muted, fontSize: 13 } },
  ], {
    x: 0.7, y: 3.8, w: 8.5, h: 0.7,
    fontFace: "Calibri Light", margin: 0,
  });

  // Bottom accent strip
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.3, w: 10, h: 0.325, fill: { color: C.accent },
  });
  s.addText("github.com/indrad3v4/Kernel-Olympics  •  endearing-rebirth.up.railway.app", {
    x: 0.5, y: 5.3, w: 9, h: 0.325,
    fontSize: 10, fontFace: "Calibri", color: "FFFFFF", align: "center",
    margin: 0,
  });
})();

// ── SLIDE 2: The $10B Problem ──
(() => {
  const s = pres.addSlide();
  s.background = { color: "F5F6FA" };
  addTitleBar(s, "The $10B Problem");

  // Left stat card
  addCard(s, 0.5, 1.1, 4.2, 3.8, "FFFFFF");

  s.addText("80%", {
    x: 0.7, y: 1.2, w: 3.8, h: 0.9,
    fontSize: 64, fontFace: "Calibri", color: C.accent, bold: true,
    margin: 0,
  });
  s.addText("hipify handles easily", {
    x: 0.7, y: 2.0, w: 3.8, h: 0.4,
    fontSize: 14, fontFace: "Calibri", color: "555555", margin: 0,
  });

  s.addText("20%", {
    x: 0.7, y: 2.8, w: 3.8, h: 0.9,
    fontSize: 64, fontFace: "Calibri", color: C.accent2, bold: true,
    margin: 0,
  });
  s.addText("Manual weeks-long engineering — warp ops, shfl, libraries", {
    x: 0.7, y: 3.6, w: 3.8, h: 0.6,
    fontSize: 13, fontFace: "Calibri", color: "666666", margin: 0,
  });

  // Right stat card
  addCard(s, 5.2, 1.1, 4.3, 3.8, "FFFFFF");

  s.addText("$10B/year", {
    x: 5.4, y: 1.2, w: 3.9, h: 0.7,
    fontSize: 36, fontFace: "Calibri", color: C.accent, bold: true,
    margin: 0,
  });
  s.addText("Enterprise migration friction cost", {
    x: 5.4, y: 1.85, w: 3.9, h: 0.4,
    fontSize: 13, fontFace: "Calibri", color: "666666", margin: 0,
  });

  s.addText("2,400+", {
    x: 5.4, y: 2.6, w: 3.9, h: 0.7,
    fontSize: 36, fontFace: "Calibri", color: C.amber, bold: true,
    margin: 0,
  });
  s.addText("CUDA kernels per average large codebase", {
    x: 5.4, y: 3.25, w: 3.9, h: 0.4,
    fontSize: 13, fontFace: "Calibri", color: "666666", margin: 0,
  });

  s.addText("MI300X outperforms NVIDIA on price/performance. Software adoption gap is the blocker.", {
    x: 5.4, y: 4.0, w: 3.9, h: 0.5,
    fontSize: 11, fontFace: "Calibri Light", color: "888888", italic: true, margin: 0,
  });

  // Bottom bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.1, w: 10, h: 0.525, fill: { color: C.bg },
  });
  s.addText("Enterprises stay on CUDA because the tail 20% takes weeks per project. Kernel Olympics kills that wait.", {
    x: 0.5, y: 5.15, w: 9, h: 0.45,
    fontSize: 11, fontFace: "Calibri Light", color: C.muted, align: "center",
    margin: 0,
  });

  addPageNum(s, 2, 9);
})();

// ── SLIDE 3: Solution Overview ──
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.08, h: 5.625, fill: { color: C.accent },
  });

  s.addText("Solution: Kernel Olympics", {
    x: 0.5, y: 0.2, w: 9, h: 0.6,
    fontSize: 26, fontFace: "Calibri", color: C.text, bold: true, margin: 0,
  });
  s.addText("A 4-LLM agentic loop that ports, validates, and verifies — no human in the middle.", {
    x: 0.5, y: 0.75, w: 9, h: 0.4,
    fontSize: 13, fontFace: "Calibri Light", color: C.muted, margin: 0,
  });

  // Flow steps
  const steps = [
    { label: "📡  SCAN", desc: "hipify + risk\nclassification", color: C.cyan },
    { label: "🧠  PLAN", desc: "DeepSeek v4\narchitectural plan", color: "2E5090" },
    { label: "⚡  PORT", desc: "GLM-5.2\nkernel code gen", color: C.accent },
    { label: "🔍  EVAL", desc: "Kimi-K2.7\n3-gate validation", color: C.amber },
    { label: "✅  VERIFY", desc: "hipcc + MI300X\nreal GPU run", color: C.green },
    { label: "📋  REPORT", desc: "Gemma\nsummary report", color: C.accent2 },
  ];

  const boxW = 1.35;
  const gap = 0.18;
  const totalW = steps.length * boxW + (steps.length - 1) * gap;
  const startX = (10 - totalW) / 2;

  steps.forEach((step, i) => {
    const x = startX + i * (boxW + gap);
    const y = 1.4;

    // Card
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: boxW, h: 2.2,
      fill: { color: step.color },
      shadow: makeShadow(),
    });

    // Label
    s.addText(step.label, {
      x, y: y + 0.15, w: boxW, h: 0.5,
      fontSize: 11, fontFace: "Calibri", color: "FFFFFF", bold: true,
      align: "center", margin: 0,
    });

    // Desc
    s.addText(step.desc, {
      x, y: y + 0.7, w: boxW, h: 1.1,
      fontSize: 10, fontFace: "Calibri Light", color: "FFFFFF",
      align: "center", valign: "top", margin: 0,
    });

    // Arrow between cards (except last)
    if (i < steps.length - 1) {
      s.addText("→", {
        x: x + boxW, y, w: gap, h: 2.2,
        fontSize: 16, color: C.muted, align: "center", valign: "middle",
        fontFace: "Calibri", margin: 0,
      });
    }
  });

  // Bottom stats row
  const stats = [
    { val: "~105s", label: "Per kernel" },
    { val: "~$0.03", label: "LLM cost" },
    { val: "30+/hr", label: "Throughput" },
    { val: "<1%", label: "False positive" },
  ];

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 4.4, w: 10, h: 1.225, fill: { color: "12121E" },
  });

  stats.forEach((stat, i) => {
    const sx = 0.8 + i * 2.3;
    s.addText(stat.val, {
      x: sx, y: 4.55, w: 2, h: 0.5,
      fontSize: 22, fontFace: "Calibri", color: C.accent, bold: true,
      align: "center", margin: 0,
    });
    s.addText(stat.label, {
      x: sx, y: 5.0, w: 2, h: 0.3,
      fontSize: 10, fontFace: "Calibri Light", color: C.muted,
      align: "center", margin: 0,
    });
  });

  addPageNum(s, 3, 9);
})();

// ── SLIDE 4: Architecture Detail ──
(() => {
  const s = pres.addSlide();
  s.background = { color: "F5F6FA" };
  addTitleBar(s, "Architecture: Multi-Agent Orchestration");

  // Main pipeline row
  const agents = [
    { name: "DeepSeek v4", role: "Planner", color: "2E5090", w: 1.6 },
    { name: "GLM-5.2", role: "Coder", color: C.accent, w: 1.4 },
    { name: "Kimi-K2.7", role: "Evaluator", color: C.amber, w: 1.6 },
    { name: "Gemma", role: "Fallback Eval", color: C.accent2, w: 1.6 },
  ];

  const startY = 1.2;
  const boxH = 1.2;
  const agentGap = 0.25;
  const totalAgentW = agents.reduce((s, a) => s + a.w, 0) + (agents.length - 1) * agentGap;
  let ax = (10 - totalAgentW) / 2;

  agents.forEach((a) => {
    const cardX = ax;
    s.addShape(pres.shapes.RECTANGLE, {
      x: cardX, y: startY, w: a.w, h: boxH,
      fill: { color: a.color },
      shadow: makeShadow(),
    });
    s.addText(a.name, {
      x: cardX, y: startY + 0.15, w: a.w, h: 0.45,
      fontSize: 13, fontFace: "Calibri", color: "FFFFFF", bold: true,
      align: "center", margin: 0,
    });
    s.addText(a.role, {
      x: cardX, y: startY + 0.6, w: a.w, h: 0.35,
      fontSize: 11, fontFace: "Calibri Light", color: "FFFFFF",
      align: "center", margin: 0,
    });
    ax += a.w + agentGap;
  });

  // Arrow labels below
  const labels = ["Plans structure", "Generates HIP", "Validates output", "Failsafe"];
  ax = (10 - totalAgentW) / 2;
  agents.forEach((a, i) => {
    s.addText(labels[i], {
      x: ax, y: startY + boxH + 0.1, w: a.w, h: 0.35,
      fontSize: 9, fontFace: "Calibri Light", color: "888888",
      align: "center", margin: 0,
    });
    ax += a.w + agentGap;
  });

  // Arrow from each agent to next
  ax = (10 - totalAgentW) / 2;
  for (let i = 0; i < agents.length - 1; i++) {
    ax += agents[i].w;
    s.addText("→", {
      x: ax, y: startY, w: agentGap, h: boxH,
      fontSize: 20, color: "999999", align: "center", valign: "middle",
      fontFace: "Calibri", margin: 0,
    });
    ax += agentGap;
  }

  // Innovation boxes below
  const innovations = [
    { title: "_strip_to_kernel_only", desc: "Pre-processor strips host code → coder sees only kernel functions. Eliminates line offset pollution from preamble." },
    { title: "Three-Gate Validation", desc: "Lexical (rejects prose) → Structural (braces/symbols) → Semantic (31 tests, <1% FP). Halts bad output before compile." },
    { title: "Pattern Memory", desc: "Trigram index + 0.2ms lookup. Cache hits skip LLM entirely — ~60,000× faster than an LLM call." },
  ];

  innovations.forEach((inn, i) => {
    const iy = 3.3;
    const ix = 0.5 + i * 3.1;
    addCard(s, ix, iy, 2.9, 1.6, "FFFFFF");

    s.addShape(pres.shapes.RECTANGLE, {
      x: ix, y: iy, w: 0.07, h: 1.6, fill: { color: C.accent },
    });

    s.addText(inn.title, {
      x: ix + 0.2, y: iy + 0.1, w: 2.6, h: 0.4,
      fontSize: 13, fontFace: "Calibri", color: "333333", bold: true, margin: 0,
    });
    s.addText(inn.desc, {
      x: ix + 0.2, y: iy + 0.5, w: 2.6, h: 0.9,
      fontSize: 10, fontFace: "Calibri Light", color: "666666", margin: 0,
    });
  });

  // Provider note
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.1, w: 10, h: 0.525, fill: { color: C.bg },
  });
  s.addText("All LLMs hosted on AMD hardware via Fireworks AI API • Runs on CPU (no GPU needed for pipeline)", {
    x: 0.5, y: 5.15, w: 9, h: 0.45,
    fontSize: 10, fontFace: "Calibri Light", color: C.muted, align: "center", margin: 0,
  });

  addPageNum(s, 4, 9);
})();

// ── SLIDE 5: Smart Porting ──
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.08, h: 5.625, fill: { color: C.accent },
  });

  s.addText("Smart Porting: _strip_to_kernel_only", {
    x: 0.5, y: 0.2, w: 9, h: 0.6,
    fontSize: 24, fontFace: "Calibri", color: C.text, bold: true, margin: 0,
  });

  // Before/After comparison
  // Before card
  s.addText("❌  Before (full CUDA source → coder)", {
    x: 0.5, y: 1.0, w: 4.3, h: 0.4,
    fontSize: 13, fontFace: "Calibri", color: C.accent2, bold: true, margin: 0,
  });
  addCard(s, 0.5, 1.4, 4.3, 3.2, "1A1A30");
  s.addText([
    { text: "// Copyright (c) 2024 NVIDIA Corporation\n// All rights reserved.\n\n#include <cuda_runtime.h>\n#include <cuda.h>\n\n__host__ void init() {\n  cudaSetDevice(0);\n}\n\n__global__ void kernel() {\n  int tid = threadIdx.x;\n  __syncthreads();\n}\n\nint main() {\n  init<<<1, 32>>>();\n  return 0;\n}", options: { fontSize: 9, color: "CCCCCC" } },
  ], {
    x: 0.7, y: 1.5, w: 3.9, h: 2.9,
    fontFace: "Consolas", margin: 0,
  });

  s.addText("⚠️ 30-line preamble → line offset pollution → false compile errors", {
    x: 0.5, y: 4.6, w: 4.3, h: 0.3,
    fontSize: 10, fontFace: "Calibri Light", color: C.accent2, align: "center", margin: 0,
  });

  // After card
  s.addText("✅  After (kernel-only → coder)", {
    x: 5.2, y: 1.0, w: 4.3, h: 0.4,
    fontSize: 13, fontFace: "Calibri", color: C.green, bold: true, margin: 0,
  });
  addCard(s, 5.2, 1.4, 4.3, 3.2, "1A1A30");
  s.addText([
    { text: "__global__ void kernel() {\n  int tid = threadIdx.x;\n  __syncthreads();\n}", options: { fontSize: 9, color: "CCCCCC" } },
  ], {
    x: 5.4, y: 1.5, w: 3.9, h: 2.2,
    fontFace: "Consolas", margin: 0,
  });

  s.addText("✓ Strips includes, host code, main() — coder sees ONLY kernel bodies", {
    x: 5.2, y: 4.6, w: 4.3, h: 0.3,
    fontSize: 10, fontFace: "Calibri Light", color: C.green, align: "center", margin: 0,
  });

  s.addText("Result: Precise line offsets → reliable HIP output → fewer retries → 2x faster pipeline", {
    x: 0.5, y: 5.15, w: 9, h: 0.35,
    fontSize: 11, fontFace: "Calibri", color: C.muted, align: "center", margin: 0,
  });

  addPageNum(s, 5, 9);
})();

// ── SLIDE 6: Three-Gate Validation ──
(() => {
  const s = pres.addSlide();
  s.background = { color: "F5F6FA" };
  addTitleBar(s, "Three-Gate Validation System");

  const gates = [
    {
      num: "01", title: "Lexical Gate", color: C.accent,
      checks: ["Rejects prose/markdown in code output", "Checks for hallucinated text blocks", "Ensures only valid CUDA/HIP syntax"],
      pass: "Pass rate: ~92%",
    },
    {
      num: "02", title: "Structural Gate", color: C.amber,
      checks: ["Braces balance verification", "Symbol table preservation check", "Function signature matching"],
      pass: "Pass rate: ~88%",
    },
    {
      num: "03", title: "Semantic Gate", color: C.green,
      checks: ["31 test cases for code correctness", "HIP API substitution validation", "<1% false-positive rate"],
      pass: "Pass rate: ~96%",
    },
  ];

  gates.forEach((g, i) => {
    const gx = 0.5 + i * 3.15;
    const gy = 1.1;
    const gw = 2.95;
    const gh = 3.3;

    addCard(s, gx, gy, gw, gh, "FFFFFF");

    // Big number
    s.addText(g.num, {
      x: gx + 0.15, y: gy + 0.1, w: 1, h: 0.5,
      fontSize: 28, fontFace: "Calibri", color: g.color, bold: true, margin: 0,
    });

    // Title
    s.addText(g.title, {
      x: gx + 0.15, y: gy + 0.55, w: gw - 0.3, h: 0.35,
      fontSize: 16, fontFace: "Calibri", color: "333333", bold: true, margin: 0,
    });

    // Checks
    const checkText = g.checks.map((c, ci) => ({
      text: c,
      options: { bullet: true, breakLine: ci < g.checks.length - 1, fontSize: 11, color: "555555" },
    }));
    s.addText(checkText, {
      x: gx + 0.15, y: gy + 1.0, w: gw - 0.3, h: 1.5,
      fontFace: "Calibri Light", margin: 0, valign: "top",
    });

    // Pass rate badge
    s.addShape(pres.shapes.RECTANGLE, {
      x: gx + 0.15, y: gy + 2.7, w: 1.6, h: 0.35,
      fill: { color: g.color },
    });
    s.addText(g.pass, {
      x: gx + 0.15, y: gy + 2.7, w: 1.6, h: 0.35,
      fontSize: 10, fontFace: "Calibri", color: "FFFFFF", align: "center", valign: "middle", margin: 0,
    });
  });

  // Bottom flow
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.7, w: 9, h: 0.65, fill: { color: C.bg },
    shadow: makeShadow(),
  });
  s.addText("Failed at any gate → auto-retry with gate error context → up to 3 retries → graceful failure report", {
    x: 0.7, y: 4.78, w: 8.6, h: 0.5,
    fontSize: 12, fontFace: "Calibri Light", color: C.muted, align: "center", valign: "middle", margin: 0,
  });

  addPageNum(s, 6, 9);
})();

// ── SLIDE 7: Performance ──
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.08, h: 5.625, fill: { color: C.accent },
  });

  s.addText("Performance & Speed", {
    x: 0.5, y: 0.2, w: 9, h: 0.6,
    fontSize: 26, fontFace: "Calibri", color: C.text, bold: true, margin: 0,
  });

  // Big stat row
  const bigStats = [
    { val: "~105s", label: "Pipeline per kernel", big: true },
    { val: "~$0.03", label: "Average cost", big: true },
    { val: "30+", label: "Kernels/hr", big: true },
  ];

  bigStats.forEach((bs, i) => {
    const bx = 0.5 + i * 3.15;
    addCard(s, bx, 1.0, 2.95, 1.4, C.card);
    s.addText(bs.val, {
      x: bx, y: 1.05, w: 2.95, h: 0.7,
      fontSize: 36, fontFace: "Calibri", color: C.accent, bold: true,
      align: "center", valign: "middle", margin: 0,
    });
    s.addText(bs.label, {
      x: bx, y: 1.75, w: 2.95, h: 0.35,
      fontSize: 12, fontFace: "Calibri Light", color: C.muted,
      align: "center", margin: 0,
    });
  });

  // Comparison vs alternatives
  s.addText("vs. Manual Engineering", {
    x: 0.5, y: 2.7, w: 3, h: 0.4,
    fontSize: 15, fontFace: "Calibri", color: C.text, bold: true, margin: 0,
  });

  const comparisons = [
    { label: "Manual (expert engineer)", time: "2–4 weeks", cost: "~$8,000", confidence: "Variable" },
    { label: "hipify + manual fix", time: "3–5 days", cost: "~$2,500", confidence: "Medium" },
    { label: "Kernel Olympics", time: "105 seconds", cost: "~$0.03", confidence: "96%" },
  ];

  // Header
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 3.2, w: 9, h: 0.4, fill: { color: C.cyan },
  });
  s.addText([
    { text: "Approach        ", options: { bold: true, fontSize: 11 } },
    { text: "Time        ", options: { bold: true, fontSize: 11 } },
    { text: "Cost          ", options: { bold: true, fontSize: 11 } },
    { text: "Confidence", options: { bold: true, fontSize: 11 } },
  ], {
    x: 0.7, y: 3.2, w: 8.6, h: 0.4,
    fontFace: "Calibri", color: "FFFFFF", valign: "middle", margin: 0,
  });

  comparisons.forEach((c, i) => {
    const rowY = 3.65 + i * 0.45;
    const rowColor = i === 2 ? "1A3A1A" : "1E1E30";
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: rowY, w: 9, h: 0.42, fill: { color: rowColor },
    });
    s.addText([
      { text: c.label.padEnd(18), options: { fontSize: 11, color: "DDDDDD" } },
      { text: c.time.padEnd(14), options: { fontSize: 11, color: i === 2 ? C.green : "AAAAAA", bold: i === 2 } },
      { text: c.cost.padEnd(16), options: { fontSize: 11, color: i === 2 ? C.green : "AAAAAA", bold: i === 2 } },
      { text: c.confidence, options: { fontSize: 11, color: i === 2 ? C.green : "AAAAAA", bold: i === 2 } },
    ], {
      x: 0.7, y: rowY, w: 8.6, h: 0.42,
      fontFace: "Consolas", valign: "middle", margin: 0,
    });
  });

  // Pattern memory shout
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 5.0, w: 9, h: 0.45, fill: { color: C.accent },
  });
  s.addText("⚡ Pattern memory: 0.2ms lookup vs ~12s LLM call = 60,000× speedup for cached patterns", {
    x: 0.7, y: 5.02, w: 8.6, h: 0.4,
    fontSize: 12, fontFace: "Calibri", color: "FFFFFF", align: "center", valign: "middle", margin: 0,
  });

  addPageNum(s, 7, 9);
})();

// ── SLIDE 8: Team ──
(() => {
  const s = pres.addSlide();
  s.background = { color: "F5F6FA" };
  addTitleBar(s, "Team Meteorite 🌠");

  const members = [
    { name: "indradev_", role: "AI Architect / Lead", desc: "Pipeline architecture, LLM orchestration, 3-gate validation, Railway deployment" },
    { name: "Aahil-Riyaz", role: "AMD/ROCm Engineering", desc: "hipcc compilation, MI300X verification, GPU testing pipeline" },
    { name: "Bromine185", role: "Kernel Engineering", desc: "CUDA kernel analysis, HIP translation patterns, unicode/edge cases" },
    { name: "meteorite67", role: "CI/CD & Testing", desc: "GitHub Actions, test suite (605+), build automation" },
    { name: "Icodemun44", role: "Infrastructure", desc: "Containerization, environment setup, dependency management" },
    { name: "_dD", role: "QA & Documentation", desc: "Debug mode, summary reports, project documentation" },
  ];

  members.forEach((m, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const mx = 0.5 + col * 3.15;
    const my = 1.1 + row * 2.0;
    const mw = 2.95;
    const mh = 1.8;

    addCard(s, mx, my, mw, mh, "FFFFFF");

    // Accent left bar
    s.addShape(pres.shapes.RECTANGLE, {
      x: mx, y: my, w: 0.06, h: mh, fill: { color: C.accent },
    });

    s.addText(m.name, {
      x: mx + 0.2, y: my + 0.1, w: mw - 0.3, h: 0.35,
      fontSize: 14, fontFace: "Calibri", color: "333333", bold: true, margin: 0,
    });
    s.addText(m.role, {
      x: mx + 0.2, y: my + 0.4, w: mw - 0.3, h: 0.3,
      fontSize: 11, fontFace: "Calibri", color: C.accent, margin: 0,
    });
    s.addText(m.desc, {
      x: mx + 0.2, y: my + 0.7, w: mw - 0.3, h: 0.9,
      fontSize: 10, fontFace: "Calibri Light", color: "666666", margin: 0,
    });
  });

  // Footer
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.1, w: 10, h: 0.525, fill: { color: C.bg },
  });
  s.addText("Built in 5 days for AMD Developer Hackathon ACT II • Team-3793", {
    x: 0.5, y: 5.15, w: 9, h: 0.45,
    fontSize: 11, fontFace: "Calibri Light", color: C.muted, align: "center", margin: 0,
  });

  addPageNum(s, 8, 9);
})();

// ── SLIDE 9: Thank You ──
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };

  // Large accent bar left
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.12, h: 5.625, fill: { color: C.accent },
  });

  s.addText("THANK YOU", {
    x: 0.7, y: 1.3, w: 8.5, h: 1.0,
    fontSize: 48, fontFace: "Calibri", color: C.text, bold: true,
    charSpacing: 6, margin: 0,
  });

  s.addText("Try Kernel Olympics — it's open source", {
    x: 0.7, y: 2.3, w: 8.5, h: 0.5,
    fontSize: 18, fontFace: "Calibri Light", color: C.muted, margin: 0,
  });

  // Links
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.7, y: 3.0, w: 4.0, h: 0.5, fill: { color: C.card },
    shadow: makeShadow(),
  });
  s.addText("🔗  github.com/indrad3v4/Kernel-Olympics", {
    x: 0.9, y: 3.0, w: 3.8, h: 0.5,
    fontSize: 13, fontFace: "Calibri", color: C.text, valign: "middle", margin: 0,
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 3.0, w: 4.0, h: 0.5, fill: { color: C.card },
    shadow: makeShadow(),
  });
  s.addText("🌐  endearing-rebirth.up.railway.app", {
    x: 5.4, y: 3.0, w: 3.8, h: 0.5,
    fontSize: 13, fontFace: "Calibri", color: C.text, valign: "middle", margin: 0,
  });

  // Q&A
  s.addText("Questions?", {
    x: 0.7, y: 3.8, w: 4, h: 0.5,
    fontSize: 20, fontFace: "Calibri", color: C.accent, bold: true, margin: 0,
  });

  // Bottom bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.3, w: 10, h: 0.325, fill: { color: C.accent },
  });
  s.addText("\"Like a meteorite, we don't arrive quietly.\"", {
    x: 0.5, y: 5.3, w: 9, h: 0.325,
    fontSize: 12, fontFace: "Calibri", color: "FFFFFF", align: "center", margin: 0,
  });
})();

// ── Write ──
pres.writeFile({ fileName: "/root/Kernel-Olympics/slides/KernelOlympics_Deck.pptx" })
  .then(() => console.log("✅ PPTX created: slides/KernelOlympics_Deck.pptx"))
  .catch(err => console.error("❌ Error:", err));
