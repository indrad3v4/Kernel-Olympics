"""
Kernel Olympics — Web Frontend (FastAPI)

Wraps the CLI pipeline as a web demo for Railway deployment.
Upload a .cu file → pipeline plans, ports, validates → returns HIP code + report.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="Kernel Olympics — CUDA→HIP Porting Pipeline")

ROOT = Path(__file__).parent.resolve()
MAIN_SCRIPT = ROOT / "src" / "main.py"
PORTED_DIR = ROOT / "ported_kernels"


@app.get("/", response_class=HTMLResponse)
async def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Kernel Olympics — CUDA to HIP Porting Pipeline</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0a0f; color: #e0e0e0; min-height: 100vh; }
    .container { max-width: 960px; margin: 0 auto; padding: 2rem; }
    header { text-align: center; margin-bottom: 3rem; }
    header h1 { font-size: 2rem; background: linear-gradient(135deg, #6c5ce7, #fd79a8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    header p { color: #888; margin-top: 0.5rem; }
    .card { background: #14141f; border: 1px solid #2a2a3a; border-radius: 12px; padding: 2rem; margin-bottom: 2rem; }
    .card h2 { color: #a78bfa; margin-bottom: 1rem; font-size: 1.2rem; }
    .drop-zone { border: 2px dashed #3a3a4a; border-radius: 8px; padding: 3rem 2rem; text-align: center; cursor: pointer; transition: all 0.3s; }
    .drop-zone:hover, .drop-zone.dragover { border-color: #6c5ce7; background: rgba(108,92,231,0.05); }
    .drop-zone.has-file { border-color: #6c5ce7; background: rgba(108,92,231,0.1); }
    .drop-zone p { color: #666; }
    .drop-zone .icon { font-size: 2rem; margin-bottom: 1rem; }
    .file-info { display: none; margin-top: 1rem; padding: 0.75rem; background: #1a1a2e; border-radius: 6px; font-size: 0.9rem; color: #a78bfa; }
    .file-info.show { display: block; }
    button { background: linear-gradient(135deg, #6c5ce7, #8b5cf6); color: white; border: none; padding: 0.75rem 2rem; border-radius: 8px; font-size: 1rem; cursor: pointer; transition: opacity 0.2s; width: 100%; margin-top: 1rem; font-weight: 600; }
    button:hover { opacity: 0.9; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .progress { display: none; margin-top: 1.5rem; }
    .progress.show { display: block; }
    .progress-bar { width: 100%; height: 4px; background: #2a2a3a; border-radius: 2px; overflow: hidden; }
    .progress-bar-fill { height: 100%; background: linear-gradient(90deg, #6c5ce7, #a78bfa); border-radius: 2px; animation: progress 2s ease-in-out infinite; width: 30%; }
    @keyframes progress { 0% { transform: translateX(-100%); } 100% { transform: translateX(430%); } }
    .progress-text { color: #888; font-size: 0.85rem; margin-top: 0.5rem; text-align: center; }
    .result { display: none; margin-top: 1.5rem; }
    .result.show { display: block; }
    .result h3 { color: #a78bfa; margin-bottom: 0.5rem; }
    pre { background: #0d0d14; border: 1px solid #2a2a3a; border-radius: 8px; padding: 1rem; overflow-x: auto; font-size: 0.8rem; line-height: 1.4; max-height: 500px; overflow-y: auto; white-space: pre-wrap; word-wrap: break-word; }
    .badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; margin-bottom: 1rem; }
    .badge.success { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
    .badge.error { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
    .badge.info { background: rgba(99,102,241,0.15); color: #6366f1; border: 1px solid rgba(99,102,241,0.3); }
    .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
    .stat { background: #0d0d14; border: 1px solid #2a2a3a; border-radius: 8px; padding: 1rem; text-align: center; }
    .stat-value { font-size: 1.5rem; font-weight: 700; color: #a78bfa; }
    .stat-label { font-size: 0.75rem; color: #666; margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .status-section { border-top: 1px solid #2a2a3a; padding-top: 1.5rem; margin-top: 1.5rem; }
    footer { text-align: center; color: #444; font-size: 0.8rem; margin-top: 3rem; padding-top: 2rem; border-top: 1px solid #1a1a2e; }
    .error-box { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); border-radius: 8px; padding: 1rem; color: #ef4444; margin-top: 1rem; display: none; }
    .error-box.show { display: block; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>⚔️ Kernel Olympics</h1>
      <p>CUDA → HIP Porting Pipeline — powered by DeepSeek · GLM · Kimi · Gemma</p>
    </header>

    <div class="card">
      <h2>📤 Upload CUDA Kernel</h2>
      <p style="color:#666; margin-bottom:1.5rem; font-size:0.9rem;">
        Drop a <code>.cu</code> file below. The pipeline plans, ports, validates, and returns HIP code.
      </p>

      <div class="drop-zone" id="dropZone">
        <div class="icon">📁</div>
        <p>Drop CUDA file here or click to browse</p>
      </div>
      <input type="file" id="fileInput" accept=".cu,.cuh,.cpp,.cuda" style="display:none">

      <div class="file-info" id="fileInfo"></div>

      <button id="portBtn" onclick="portCode()">🚀 Port to HIP</button>
    </div>

    <div class="progress" id="progress">
      <div class="progress-bar"><div class="progress-bar-fill"></div></div>
      <div class="progress-text">Porting kernel... this takes 1–3 minutes</div>
    </div>

    <div class="error-box" id="errorBox"></div>

    <div class="result" id="result">
      <div class="card">
        <h2>📊 Pipeline Report</h2>
        <div class="badge info" id="statusBadge">Complete</div>
        <div class="stats" id="stats"></div>
        <div class="status-section">
          <h3>🔄 Gate Results</h3>
          <div id="gates"></div>
        </div>
      </div>
      <div class="card">
        <h2>📝 Ported HIP Code</h2>
        <pre id="hipOutput"></pre>
      </div>
    </div>

    <footer>
      Kernel Olympics · Nous Research · AMD ROCm Hackathon 2026
    </footer>
  </div>

  <script>
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    const portBtn = document.getElementById('portBtn');
    const progress = document.getElementById('progress');
    const result = document.getElementById('result');
    const errorBox = document.getElementById('errorBox');
    const hipOutput = document.getElementById('hipOutput');
    const stats = document.getElementById('stats');
    const gates = document.getElementById('gates');

    dropZone.onclick = () => fileInput.click();
    dropZone.ondragover = (e) => { e.preventDefault(); dropZone.classList.add('dragover'); };
    dropZone.ondragleave = () => dropZone.classList.remove('dragover');
    dropZone.ondrop = (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); handleFile(e.dataTransfer.files[0]); };
    fileInput.onchange = () => handleFile(fileInput.files[0]);

    function handleFile(file) {
      if (!file) return;
      const ext = file.name.split('.').pop();
      if (!['cu', 'cuh', 'cpp', 'cuda', 'txt'].includes(ext)) {
        showError('Please upload a .cu (CUDA kernel) file');
        return;
      }
      dropZone.classList.add('has-file');
      fileInfo.textContent = '📄 ' + file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
      fileInfo.classList.add('show');
      portBtn.disabled = false;
    }

    function showError(msg) {
      errorBox.textContent = msg;
      errorBox.classList.add('show');
      setTimeout(() => errorBox.classList.remove('show'), 5000);
    }

    async function portCode() {
      portBtn.disabled = true;
      progress.classList.add('show');
      result.classList.remove('show');
      errorBox.classList.remove('show');

      const file = fileInput.files[0];
      if (!file) {
        showError('Upload a .cu file first');
        portBtn.disabled = false;
        progress.classList.remove('show');
        return;
      }

      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await fetch('/port', { method: 'POST', body: formData });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Porting failed');

        const elapsed = data.report?.elapsed_seconds || 'N/A';
        const cost = data.report?.cost || 'N/A';
        const verdict = data.report?.verdict || 'N/A';
        const iterations = data.report?.iterations || 'N/A';

        stats.innerHTML = '' +
          '<div class="stat"><div class="stat-value">' + elapsed + 's</div><div class="stat-label">Runtime</div></div>' +
          '<div class="stat"><div class="stat-value">$' + cost + '</div><div class="stat-label">Cost</div></div>' +
          '<div class="stat"><div class="stat-value">' + iterations + '</div><div class="stat-label">Iterations</div></div>' +
          '<div class="stat"><div class="stat-value">' + verdict + '</div><div class="stat-label">Status</div></div>';

        const gatesList = data.gate_results || [];
        gates.innerHTML = gatesList.length
          ? gatesList.map(function(g) {
              return '<div style="padding:0.5rem; margin:0.25rem 0; background:#0d0d14; border-radius:4px; border-left:3px solid ' + (g.passed ? '#10b981' : '#ef4444') + '"><span style="color:' + (g.passed ? '#10b981' : '#ef4444') + '">' + (g.passed ? '\u2713' : '\u2717') + '</span> ' + g.name + ': ' + g.message + '</div>';
            }).join('')
          : '<p style="color:#666; font-size:0.85rem;">No gate results recorded</p>';

        hipOutput.textContent = data.hip_code || 'No HIP code generated';
        result.classList.add('show');
      } catch (err) {
        showError(err.message);
      } finally {
        portBtn.disabled = false;
        progress.classList.remove('show');
      }
    }
  </script>
</body>
</html>"""


@app.post("/port")
async def port_file(file: UploadFile = File(...)):
    """Accept a CUDA file, run the pipeline, return HIP code + report."""
    # Save uploaded file to a temp location
    suffix = Path(file.filename).suffix or ".cu"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="wb") as f:
        content = await file.read()
        f.write(content)
        temp_path = f.name

    # Temp path for pipeline JSON report
    report_fd, report_path = tempfile.mkstemp(suffix=".json")
    os.close(report_fd)

    t0 = time.time()
    try:
        # Run the pipeline — it writes the report JSON to --output
        result = subprocess.run(
            [sys.executable, str(MAIN_SCRIPT), "--input", temp_path,
             "--output", report_path],
            capture_output=True,
            text=True,
            timeout=600,  # 10-minute hard cap
            cwd=str(ROOT),
        )

        elapsed = round(time.time() - t0, 1)

        # Read the JSON report from the output file
        report = {}
        if os.path.exists(report_path):
            try:
                with open(report_path, encoding="utf-8") as rf:
                    report = json.load(rf)
            except (json.JSONDecodeError, OSError) as e:
                report = {"error": f"Failed to parse report: {e}"}

        # Find the ported kernel — try report first, then fall back to disk
        hip_code = ""
        kernel_name = Path(file.filename).stem

        # Read ported_code from report's verification results
        sections = report.get("sections", {})
        verifications = sections.get("verification", [])
        for vr in verifications:
            if vr.get("kernel") == kernel_name:
                hip_code = vr.get("ported_code", "")
                break

        # Fallback: read from ported_kernels/ on disk
        if not hip_code:
            ported_file = PORTED_DIR / f"{kernel_name}.hip.cpp"
            if ported_file.exists():
                hip_code = ported_file.read_text(encoding="utf-8")

        # Extract pipeline state
        ps = report.get("pipeline_state", {})
        verdict = report.get("result", ps.get("result", "N/A"))
        cost = ps.get("total_cost", 0)
        cache_hits = ps.get("cache_hits", 0)
        llm_calls = ps.get("llm_calls", 0)
        iterations = ps.get("iterations_used", "N/A")

        return {
            "hip_code": hip_code,
            "report": {
                "elapsed_seconds": elapsed,
                "cost": f"{cost:.4f}",
                "verdict": verdict,
                "iterations": iterations,
                "cache_hits": cache_hits,
                "llm_calls": llm_calls,
            },
            "gate_results": verifications[0].get("gate_results", []) if verifications else [],
            "pipeline_output": result.stdout[-2000:] if result.stdout else "",
        }

    except subprocess.TimeoutExpired:
        return JSONResponse(
            status_code=408,
            content={"detail": "Pipeline timed out after 600 seconds"},
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
    finally:
        # Cleanup temp files
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        try:
            os.unlink(report_path)
        except OSError:
            pass
