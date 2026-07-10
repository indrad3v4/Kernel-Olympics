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
  textDark: "222222", // dark text for light bgs
  muted:   "8899AA",  // muted gray-blue
  cyan:    "0F3460",  // dark cyan
  green:   "4ECB71",
  amber:   "FFB347",
};

// ── Helpers ──
function addFooterBar(s, text) {
  pres.addSlide().addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.2, w: 10, h: 0.425, fill: { color: C.bg },
  });
}

function addPageNum(s, num, total) {
  s.addText(`${num} / ${total}`, {
    x: 8.5, y: 5.2, w: 1.2, h: 0.35,
    fontSize: 9, color: C.muted, fontFace: "Calibri", align: "right",
  });
}

function addCard(s, x, y, w, h, fillColor) {
  s.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: fillColor || C.card },
    shadow: { type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.25 },
  });
}

function addTitleBar(s, text) {
  // Red accent line at top
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.08,
    fill: { color: C.accent },
  });
  // Title text in DARK color for light background slides
  s.addText(text, {
    x: 0.5, y: 0.2, w: 9, h: 0.6,
    fontSize: 22, fontFace: "Calibri", color: C.textDark, bold: true,
    margin: 0,
  });
}

// ── SLIDE 1: Title ──
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.12, h: 5.625, fill: { color: C.accent },
  });

  s.addText("KERNEL OLYMPICS", {
    x: 0.7, y: 0.9, w: 8.5, h: 1.2,
    fontSize: 44, fontFace: "Calibri", color: C.text, bold: true,
    charSpacing: 4, margin: 0,
  });

  s.addText("CUDA → ROCm Migration Copilot", {
    x: 0.7, y: 2.1, w: 8.5, h: 0.7,
    fontSize: 24, fontFace: "Calibri", color: C.accent, bold: false, margin: 0,
  });

  s.addText("Ship AMD-ready code in minutes, not months.", {
    x: 0.7, y: 2.8, w: 8.5, h: 0.5,
    fontSize: 16, fontFace: "Calibri Light", color: C.muted, margin: 0,
  });

  s.addShape(pres.shapes.LINE, {
    x: 0.7, y: 3.45, w: 3.0, h: 0,
    line: { color: C.accent, width: 2 },
  });

  s.addText("AMD Developer Hackathon ACT II  •  Track 3 — Unicorn\nTeam Meteorite 🌠  •  Team-3793", {
    x: 0.7, y: 3.7, w: 8.5, h: 0.6,
    fontSize: 13, fontFace: "Calibri Light", color: C.muted, margin: 0,
  });

  // Footer bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.2, w: 10, h: 0.425, fill: { color: C.accent },
  });
  s.addText("github.com/indrad3v4/Kernel-Olympics  •  endearing-rebirth.up.railway.app", {
    x: 0.3, y: 5.22, w: 9.4, h: 0.4,
    fontSize: 10, fontFace: "Calibri", color: "FFFFFF", align: "center", valign: "middle", margin: 0,
  });
})();

// ── SLIDE 2: The $10B Problem ──
(() => {
  const s = pres.addSlide();
  s.background = { color: "F5F6FA" };
  addTitleBar(s, "The $10B Problem");

  addCard(s, 0.5, 1.1, 4.2, 3.6, "FFFFFF");

  s.addText("80%", {
    x: 0.7, y: 1.2, w: 3.8, h: 0.8,
    fontSize: 64, fontFace: "Calibri", color: C.accent, bold: true, margin: 0,
  });
  s.addText("hipify handles easily", {
    x: 0.7, y: 1.95, w: 3.8, h: 0.4,
    fontSize: 14, fontFace: "Calibri", color: "555555", margin: 0,
  });

  s.addText("20%", {
    x: 0.7, y: 2.5, w: 3.8, h: 0.8,
    fontSize: 64, fontFace: "Calibri", color: C.accent2, bold: true, margin: 0,
  });
  s.addText("Manual weeks-long engineering — warp ops, shfl, libraries", {
    x: 0.7, y: 3.3, w: 3.8, h: 0.6,
    fontSize: 13, fontFace: "Calibri", color: "666666", margin: 0,
  });

  addCard(s, 5.2, 1.1, 4.3, 3.6, "FFFFFF");

  s.addText("$10B/year", {
    x: 5.4, y: 1.2, w: 3.9, h: 0.7,
    fontSize: 36, fontFace: "Calibri", color: C.accent, bold: true, margin: 0,
  });
  s.addText("Enterprise migration friction cost", {
    x: 5.4, y: 1.85, w: 3.9, h: 0.35,
    fontSize: 13, fontFace: "Calibri", color: "666666", margin: 0,
  });

  s.addText("2,400+", {
    x: 5.4, y: 2.5, w: 3.9, h: 0.7,
    fontSize: 36, fontFace: "Calibri", color: C.amber, bold: true, margin: 0,
  });
  s.addText("CUDA kernels per large codebase", {
    x: 5.4, y: 3.15, w: 3.9, h: 0.35,
    fontSize: 13, fontFace: "Calibri", color: "666666", margin: 0,
  });

  s.addText("MI300X outperforms NVIDIA on price/performance. Software adoption gap is the blocker.", {
    x: 5.4, y: 3.8, w: 3.9, h: 0.5,
    fontSize: 11, fontFace: "Calibri Light", color: "888888", italic: true, margin: 0,
  });

  // Bottom bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.0, w: 10, h: 0.625, fill: { color: C.bg },
  });
  s.addText("Enterprises stay on CUDA because the tail 20% takes weeks per project. Kernel Olympics kills that wait.", {
    x: 0.5, y: 5.1, w: 9, h: 0.45,
    fontSize: 11, fontFace: "Calibri Light", color: C.muted, align: "center", valign: "middle", margin: 0,
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

  const steps = [
    { label: "SCAN", icon: "\uD83D\uDEE1", desc: "hipify + risk\nclassification", color: C.cyan },
    { label: "PLAN", icon: "\uD83E\uDDE0", desc: "DeepSeek v4\narchitectural plan", color: "2E5090" },
    { label: "PORT", icon: "\u26A1", desc: "GLM-5.2\nkernel code gen", color: C.accent },
    { label: "EVAL", icon: "\uD83D\uDD0D", desc: "Kimi-K2.7\n3-gate validation", color: C.amber },
    { label: "VERIFY", icon: "\u2705", desc: "hipcc + MI300X\nreal GPU run", color: C.green },
    { label: "REPORT", icon: "\uD83D\uDCCB", desc: "Gemma\nsummary report", color: C.accent2 },
  ];

  const boxW = 1.35;
  const gap = 0.18;
  const totalW = steps.length * boxW + (steps.length - 1) * gap;
  const startX = (10 - totalW) / 2;

  steps.forEach((step, i) => {
    const x = startX + i * (boxW + gap);
    const y = 1.4;

    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: boxW, h: 2.2, fill: { color: step.color },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.3 },
    });

    s.addText(`${step.icon}  ${step.label}`, {
      x, y: y + 0.1, w: boxW, h: 0.5,
      fontSize: 12, fontFace: "Calibri", color: "FFFFFF", bold: true,
      align: "center", margin: 0,
    });

    s.addText(step.desc, {
      x, y: y + 0.65, w: boxW, h: 1.0,
      fontSize: 10, fontFace: "Calibri Light", color: "FFFFFF",
      align: "center", valign: "top", margin: 0,
    });

    if (i < steps.length - 1) {
      s.addText("\u2192", {
        x: x + boxW, y, w: gap, h: 2.2,
        fontSize: 16, color: C.muted, align: "center", valign: "middle",
        fontFace: "Calibri", margin: 0,
      });
    }
  });

  // Bottom stats
  const stats = [
    { val: "~105s", label: "Per kernel" },
    { val: "~$0.03", label: "LLM cost" },
    { val: "30+/hr", label: "Throughput" },
    { val: "<1%", label: "False positive" },
  ];

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 4.3, w: 10, h: 1.325, fill: { color: "0C0C18" },
  });

  stats.forEach((stat, i) => {
    const sx = 0.8 + i * 2.3;
    s.addText(stat.val, {
      x: sx, y: 4.45, w: 2, h: 0.5,
      fontSize: 24, fontFace: "Calibri", color: C.accent, bold: true,
      align: "center", margin: 0,
    });
    s.addText(stat.label, {
      x: sx, y: 4.95, w: 2, h: 0.3,
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

  const arrowLabels = ["Plans structure", "Generates HIP", "Validates output", "Failsafe"];

  agents.forEach((a, i) => {
    const cardX = ax;
    s.addShape(pres.shapes.RECTANGLE, {
      x: cardX, y: startY, w: a.w, h: boxH,
      fill: { color: a.color },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.25 },
    });
    s.addText(a.name, {
      x: cardX, y: startY + 0.15, w: a.w, h: 0.45,
      fontSize: 14, fontFace: "Calibri", color: "FFFFFF", bold: true,
      align: "center", margin: 0,
    });
    s.addText(a.role, {
      x: cardX, y: startY + 0.6, w: a.w, h: 0.35,
      fontSize: 11, fontFace: "Calibri Light", color: "FFFFFF",
      align: "center", margin: 0,
    });
    s.addText(arrowLabels[i], {
      x: cardX, y: startY + boxH + 0.1, w: a.w, h: 0.35,
      fontSize: 10, fontFace: "Calibri Light", color: "777777",
      align: "center", margin: 0,
    });
    ax += a.w + agentGap;
  });

  // Arrows between agents
  ax = (10 - totalAgentW) / 2;
  for (let i = 0; i < agents.length - 1; i++) {
    ax += agents[i].w;
    s.addText("\u2192", {
      x: ax, y: startY, w: agentGap, h: boxH,
      fontSize: 20, color: "999999", align: "center", valign: "middle",
      fontFace: "Calibri", margin: 0,
    });
    ax += agentGap;
  }

  // Innovation boxes
  const innovations = [
    { title: "_strip_to_kernel_only", desc: "Pre-processor strips host code \u2192 coder sees only kernel functions. Eliminates line offset pollution from preamble." },
    { title: "Three-Gate Validation", desc: "Lexical (rejects prose) \u2192 Structural (braces/symbols) \u2192 Semantic (31 tests, <1% FP). Halts bad output before compile." },
    { title: "Pattern Memory", desc: "Trigram index + 0.2ms lookup. Cache hits skip LLM entirely \u2014 ~60,000\u00d7 faster than an LLM call." },
  ];

  innovations.forEach((inn, i) => {
    const iy = 3.2;
    const ix = 0.5 + i * 3.1;
    addCard(s, ix, iy, 2.9, 1.65, "FFFFFF");

    s.addShape(pres.shapes.RECTANGLE, {
      x: ix, y: iy, w: 0.07, h: 1.65, fill: { color: C.accent },
    });

    s.addText(inn.title, {
      x: ix + 0.2, y: iy + 0.1, w: 2.6, h: 0.4,
      fontSize: 13, fontFace: "Calibri", color: C.textDark, bold: true, margin: 0,
    });
    s.addText(inn.desc, {
      x: ix + 0.2, y: iy + 0.55, w: 2.6, h: 1.0,
      fontSize: 10, fontFace: "Calibri Light", color: "555555", margin: 0,
    });
  });

  // Provider note bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.1, w: 10, h: 0.525, fill: { color: C.bg },
  });
  s.addText("All LLMs hosted on AMD hardware via Fireworks AI API \u2022 Pipeline runs on CPU (no GPU needed for orchestration)", {
    x: 0.5, y: 5.15, w: 9, h: 0.45,
    fontSize: 10, fontFace: "Calibri Light", color: C.muted, align: "center", valign: "middle", margin: 0,
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

  // Before
  s.addText("Before (full CUDA source \u2192 coder)", {
    x: 0.5, y: 1.0, w: 4.3, h: 0.4,
    fontSize: 13, fontFace: "Calibri", color: C.accent2, bold: true, margin: 0,
  });
  addCard(s, 0.5, 1.4, 4.3, 3.0, "1A1A30");
  s.addText(
    "// Copyright (c) 2024 NVIDIA Corp\n// All rights reserved.\n\n#include <cuda_runtime.h>\n\n__host__ void init() {\n  cudaSetDevice(0);\n}\n\n__global__ void kernel() {\n  int tid = threadIdx.x;\n  __syncthreads();\n}\n\nint main() {\n  init<<<1, 32>>>();\n  return 0;\n}", {
    x: 0.7, y: 1.5, w: 3.9, h: 2.7,
    fontSize: 9, fontFace: "Consolas", color: "CCCCCC", margin: 0,
  });

  s.addText("30-line preamble \u2192 line offset pollution \u2192 false compile errors", {
    x: 0.5, y: 4.45, w: 4.3, h: 0.3,
    fontSize: 10, fontFace: "Calibri Light", color: C.accent2, align: "center", margin: 0,
  });

  // After
  s.addText("After (kernel-only \u2192 coder)", {
    x: 5.2, y: 1.0, w: 4.3, h: 0.4,
    fontSize: 13, fontFace: "Calibri", color: C.green, bold: true, margin: 0,
  });
  addCard(s, 5.2, 1.4, 4.3, 3.0, "1A1A30");
  s.addText(
    "__global__ void kernel() {\n  int tid = threadIdx.x;\n  __syncthreads();\n}", {
    x: 5.4, y: 1.5, w: 3.9, h: 2.0,
    fontSize: 9, fontFace: "Consolas", color: "CCCCCC", margin: 0,
  });

  s.addText("Strips includes, host code, main() \u2014 coder sees ONLY kernel bodies", {
    x: 5.2, y: 4.45, w: 4.3, h: 0.3,
    fontSize: 10, fontFace: "Calibri Light", color: C.green, align: "center", margin: 0,
  });

  s.addText("Result: Precise line offsets \u2192 reliable HIP output \u2192 fewer retries \u2192 2x faster pipeline", {
    x: 0.5, y: 5.0, w: 9, h: 0.35,
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
    const gh = 3.5;

    addCard(s, gx, gy, gw, gh, "FFFFFF");

    s.addText(g.num, {
      x: gx + 0.15, y: gy + 0.1, w: 1, h: 0.5,
      fontSize: 28, fontFace: "Calibri", color: g.color, bold: true, margin: 0,
    });
    s.addText(g.title, {
      x: gx + 0.15, y: gy + 0.55, w: gw - 0.3, h: 0.35,
      fontSize: 16, fontFace: "Calibri", color: C.textDark, bold: true, margin: 0,
    });

    const checkText = g.checks.map((c, ci) => ({
      text: c,
      options: { bullet: true, breakLine: ci < g.checks.length - 1, fontSize: 11, color: "555555" },
    }));
    s.addText(checkText, {
      x: gx + 0.15, y: gy + 1.0, w: gw - 0.3, h: 1.5,
      fontFace: "Calibri Light", margin: 0, valign: "top",
    });

    s.addShape(pres.shapes.RECTANGLE, {
      x: gx + 0.15, y: gy + 2.7, w: 1.6, h: 0.35,
      fill: { color: g.color },
    });
    s.addText(g.pass, {
      x: gx + 0.15, y: gy + 2.7, w: 1.6, h: 0.35,
      fontSize: 10, fontFace: "Calibri", color: "FFFFFF", align: "center", valign: "middle", margin: 0,
    });
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.85, w: 9, h: 0.65, fill: { color: C.bg },
    shadow: { type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.25 },
  });
  s.addText("Failed at any gate \u2192 auto-retry with gate error context \u2192 up to 3 retries \u2192 graceful failure report", {
    x: 0.7, y: 4.92, w: 8.6, h: 0.5,
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
    { val: "~105s", label: "Per kernel pipeline" },
    { val: "~$0.03", label: "Average LLM cost" },
    { val: "30+/hr", label: "Throughput" },
  ];

  bigStats.forEach((bs, i) => {
    const bx = 0.5 + i * 3.15;
    addCard(s, bx, 1.0, 2.95, 1.4, C.card);
    s.addText(bs.val, {
      x: bx, y: 1.05, w: 2.95, h: 0.7,
      fontSize: 38, fontFace: "Calibri", color: C.accent, bold: true,
      align: "center", valign: "middle", margin: 0,
    });
    s.addText(bs.label, {
      x: bx, y: 1.75, w: 2.95, h: 0.35,
      fontSize: 12, fontFace: "Calibri Light", color: C.muted,
      align: "center", margin: 0,
    });
  });

  // Comparison table
  s.addText("vs. Manual Engineering", {
    x: 0.5, y: 2.7, w: 9, h: 0.35,
    fontSize: 15, fontFace: "Calibri", color: C.text, bold: true, margin: 0,
  });

  const comparisons = [
    { label: "Manual (expert engineer)",    time: "2\u20134 weeks", cost: "$8,000",     conf: "Variable" },
    { label: "hipify + manual fix",        time: "3\u20135 days",   cost: "$2,500",     conf: "Medium"  },
    { label: "Kernel Olympics",           time: "105s",      cost: "$0.03",     conf: "96%"   },
  ];

  // Table header
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 3.15, w: 9, h: 0.4, fill: { color: C.cyan },
  });
  s.addText("  Approach                     Time            Cost             Confidence", {
    x: 0.7, y: 3.15, w: 8.6, h: 0.4,
    fontSize: 11, fontFace: "Calibri", color: "FFFFFF", bold: true, valign: "middle", margin: 0,
  });

  comparisons.forEach((c, i) => {
    const rowY = 3.6 + i * 0.5;
    const rowColor = i === 2 ? "1A3A1A" : "1E1E30";
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: rowY, w: 9, h: 0.46, fill: { color: rowColor },
    });
    const clr = i === 2 ? C.green : "AAAAAA";
    const bld = i === 2;
    s.addText(c.label, {
      x: 0.7, y: rowY, w: 3.2, h: 0.46,
      fontSize: 11, fontFace: "Calibri", color: "DDDDDD", bold: bld, valign: "middle", margin: 0,
    });
    s.addText(c.time, {
      x: 3.9, y: rowY, w: 1.5, h: 0.46,
      fontSize: 11, fontFace: "Calibri", color: clr, bold: bld, valign: "middle", margin: 0,
    });
    s.addText(c.cost, {
      x: 5.4, y: rowY, w: 1.5, h: 0.46,
      fontSize: 11, fontFace: "Calibri", color: clr, bold: bld, valign: "middle", margin: 0,
    });
    s.addText(c.conf, {
      x: 7.2, y: rowY, w: 1.0, h: 0.46,
      fontSize: 11, fontFace: "Calibri", color: clr, bold: bld, valign: "middle", margin: 0,
    });
  });

  // Pattern memory shoutout
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 5.1, w: 9, h: 0.45, fill: { color: C.accent },
  });
  s.addText("Pattern memory: 0.2ms lookup vs ~12s LLM call = 60,000\u00d7 speedup for cached patterns", {
    x: 0.7, y: 5.12, w: 8.6, h: 0.4,
    fontSize: 12, fontFace: "Calibri", color: "FFFFFF", align: "center", valign: "middle", margin: 0,
  });

  addPageNum(s, 7, 9);
})();

// ── SLIDE 8: Team ──
(() => {
  const s = pres.addSlide();
  s.background = { color: "F5F6FA" };
  addTitleBar(s, "Team Meteorite \uD83C\uDF20");

  const members = [
    { name: "indradev_", role: "AI Architect / Lead", desc: "Pipeline architecture, LLM orchestration, 3-gate validation, Railway deploy" },
    { name: "Aahil-Riyaz", role: "AMD/ROCm Engineering", desc: "hipcc compilation, MI300X verification, GPU testing pipeline" },
    { name: "Bromine185", role: "Kernel Engineering", desc: "CUDA kernel analysis, HIP translation patterns, unicode/edge cases" },
    { name: "meteorite67", role: "CI/CD & Testing", desc: "GitHub Actions, test suite (605+), build automation" },
    { name: "Icodemun44", role: "Infrastructure", desc: "Containerization, environment setup, dependency management" },
    { name: "_dD", role: "QA & Documentation", desc: "Debug mode, summary reports, project docs" },
  ];

  members.forEach((m, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const mx = 0.5 + col * 3.15;
    const my = 1.1 + row * 2.0;
    const mw = 2.95;
    const mh = 1.8;

    addCard(s, mx, my, mw, mh, "FFFFFF");

    s.addShape(pres.shapes.RECTANGLE, {
      x: mx, y: my, w: 0.06, h: mh, fill: { color: C.accent },
    });

    s.addText(m.name, {
      x: mx + 0.2, y: my + 0.1, w: mw - 0.3, h: 0.35,
      fontSize: 14, fontFace: "Calibri", color: C.textDark, bold: true, margin: 0,
    });
    s.addText(m.role, {
      x: mx + 0.2, y: my + 0.4, w: mw - 0.3, h: 0.3,
      fontSize: 11, fontFace: "Calibri", color: C.accent, margin: 0,
    });
    s.addText(m.desc, {
      x: mx + 0.2, y: my + 0.7, w: mw - 0.3, h: 0.9,
      fontSize: 10, fontFace: "Calibri Light", color: "555555", margin: 0,
    });
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.1, w: 10, h: 0.525, fill: { color: C.bg },
  });
  s.addText("Built in 5 days for AMD Developer Hackathon ACT II \u2022 Team-3793", {
    x: 0.3, y: 5.15, w: 9.4, h: 0.45,
    fontSize: 11, fontFace: "Calibri Light", color: C.muted, align: "center", valign: "middle", margin: 0,
  });

  addPageNum(s, 8, 9);
})();

// ── SLIDE 9: Thank You ──
(() => {
  const s = pres.addSlide();
  s.background = { color: C.bg };

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.12, h: 5.625, fill: { color: C.accent },
  });

  s.addText("THANK YOU", {
    x: 0.7, y: 1.1, w: 8.5, h: 1.0,
    fontSize: 48, fontFace: "Calibri", color: C.text, bold: true,
    charSpacing: 6, margin: 0,
  });

  s.addText("Try Kernel Olympics \u2014 it's open source", {
    x: 0.7, y: 2.1, w: 8.5, h: 0.5,
    fontSize: 18, fontFace: "Calibri Light", color: C.muted, margin: 0,
  });

  // Link buttons
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.7, y: 2.8, w: 4.0, h: 0.5, fill: { color: C.card },
    shadow: { type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.3 },
  });
  s.addText("github.com/indrad3v4/Kernel-Olympics", {
    x: 0.9, y: 2.8, w: 3.8, h: 0.5,
    fontSize: 13, fontFace: "Calibri", color: C.text, valign: "middle", margin: 0,
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 2.8, w: 4.0, h: 0.5, fill: { color: C.card },
    shadow: { type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.3 },
  });
  s.addText("endearing-rebirth.up.railway.app", {
    x: 5.4, y: 2.8, w: 3.8, h: 0.5,
    fontSize: 13, fontFace: "Calibri", color: C.text, valign: "middle", margin: 0,
  });

  s.addText("Questions?", {
    x: 0.7, y: 3.7, w: 4, h: 0.5,
    fontSize: 20, fontFace: "Calibri", color: C.accent, bold: true, margin: 0,
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.2, w: 10, h: 0.425, fill: { color: C.accent },
  });
  s.addText("\"Like a meteorite, we don't arrive quietly.\"", {
    x: 0.3, y: 5.22, w: 9.4, h: 0.4,
    fontSize: 12, fontFace: "Calibri", color: "FFFFFF", align: "center", valign: "middle", margin: 0,
  });
})();

// ── Write ──
pres.writeFile({ fileName: "/root/Kernel-Olympics/slides/KernelOlympics_Deck.pptx" })
  .then(() => console.log("PPTX created: slides/KernelOlympics_Deck.pptx"))
  .catch(err => console.error("Error:", err));
