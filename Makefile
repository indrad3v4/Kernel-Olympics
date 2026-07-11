# ──────────────────────────────────────────────────────────────
# Kernel Olympics — Makefile
# CUDA→ROCm migration pipeline via 4‑LLM multi‑agent loop
#
# Designed to read like a Python module docstring.
# Every target is self‑documenting (make help shows all).
#
# ──────────────────────────────────────────────────────────────

.DEFAULT_GOAL := help
SHELL         := /usr/bin/env bash
PYTHON        := python3
CU_FILE       ?= sample_kernels/cuda/nvidia_shfl_scan.cu
KERNEL_NAME   ?= $(shell basename "$(CU_FILE)" .cu)
PORTED        := ported_kernels/$(KERNEL_NAME).hip.cpp
BINARY        := /tmp/$(KERNEL_NAME)_proof
PROOF         := /tmp/proof_harness.hip.cpp
GH_USER       ?= indrad3v4
GH_REPO       ?= Kernel-Olympics
GH_BRANCH     ?= main

# ── Virtual environment ---------------------------------------
.venv:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --quiet --upgrade pip setuptools wheel

venv: .venv  ## Create Python virtual environment
	@echo "  ✅ .venv ready — activate with:  source .venv/bin/activate"

install: .venv  ## Install all Python dependencies (editable)
	.venv/bin/pip install --quiet -e "."
	@echo "  ✅ Dependencies installed"

# ── Pipeline targets ------------------------------------------

port: .venv  ## Run pipeline (CUDA→HIP) on one kernel:  make port CU_FILE=path.cu
	$(PYTHON) -m src.main --input "$(CU_FILE)"

port-all: .venv  ## Run pipeline on ALL sample kernels at once
	$(PYTHON) -m src.main --input sample_kernels/cuda/*.cu

# ── All‑in‑one -------------------------------------------------

pipeline: port compile run  ## Full cycle: CUDA → ROCm in one command
	@echo "  ✅ $(CU_FILE) → AMD GPU — DONE"

# ── Compile + Run helpers (separate targets for clarity) ------

compile: $(PORTED)  ## hipcc device‑only proof harness and compile
	@scripts/generate_proof.py "$(PORTED)" "$(KERNEL_NAME)" "$(PROOF)"
	hipcc -o $(BINARY) $(PROOF) -I/opt/rocm/include 2>&1 | sed 's/^/  │ /'
	@echo "  ✅ Compiled → $(BINARY)"

run: $(BINARY)  ## Run compiled proof on AMD GPU
	$(BINARY)

$(PORTED):
	@echo "  ⚠  Ported kernel not found — run  make port  first"
	@exit 1

$(BINARY):
	@echo "  ⚠  Binary not found — run  make compile  first"
	@exit 1

# ── Demo -------------------------------------------------------

demo:  ## Run the full demo (migration + compile + proof)
	bash scripts/demo_recording.sh run

record:  ## Record the demo with asciinema
	bash scripts/demo_recording.sh record

# ── Quality ----------------------------------------------------

test: .venv  ## Run test suite (665 tests)
	$(PYTHON) -m pytest tests/ -x -q

testv: .venv  ## Run tests verbosely
	$(PYTHON) -m pytest tests/ -x -v

lint:  ## Run ruff linter
	ruff check src/ tests/ && ruff format --check src/ tests/

doctor:  ## Pre‑flight environment check
	$(PYTHON) -m src.main --doctor

# ── Cache / memory ---------------------------------------------

fresh: .venv  ## Run with empty pattern memory (fresh cache)
	$(PYTHON) -m src.main --input "$(CU_FILE)" --fresh

speed-demo: .venv  ## Run speed‑comparison demo: LLM vs cache
	$(PYTHON) -m src.main --demo

speed-demo-reset: .venv  ## Speed demo with fresh cache
	$(PYTHON) -m src.main --demo --reset

# ── GitHub sync ------------------------------------------------

pull:  ## git pull latest from GitHub
	git pull origin $(GH_BRANCH)
	@echo "  ✅ $(GH_BRANCH) synced from github.com/$(GH_USER)/$(GH_REPO)"

push:  ## git add -A + commit + push (use MSG="message")
	git add -A && git commit -m "$(MSG)" && git push origin $(GH_BRANCH)
	@echo "  ✅ Pushed to github.com/$(GH_USER)/$(GH_REPO)"

sync: pull push  ## Pull latest, then push local changes

# ── First-time setup ------------------------------------------

setup:  ## Clone repo + venv + install (one-shot on new machine)
	git clone https://github.com/$(GH_USER)/$(GH_REPO).git /tmp/$(GH_REPO) 2>/dev/null; \
	if [ -d /tmp/$(GH_REPO) ]; then \
	    rsync -a /tmp/$(GH_REPO)/ .; rm -rf /tmp/$(GH_REPO); \
	fi; \
	git pull origin $(GH_BRANCH)
	$(MAKE) install
	@echo "  ✅ $(GH_REPO) ready on this machine"

reclone:  ## Full re-clone (zaps local changes)
	rm -rf .git .venv
	git clone https://github.com/$(GH_USER)/$(GH_REPO).git /tmp/$(GH_REPO) && \
	rsync -a --remove-source-files /tmp/$(GH_REPO)/ . && \
	rm -rf /tmp/$(GH_REPO) && \
	$(MAKE) install
	@echo "  ✅ Fresh clone of $(GH_REPO)"

# ── Deployment -------------------------------------------------

push-to-cation: pull  ## Push to AMD workspace (cation) via SSH
	ssh cation "cd /workspace/Kernel-Olympics && git pull origin $(GH_BRANCH)"
	@echo "  ✅ cation synced to origin/main"

# ── Debug / daemon ---------------------------------------------

debug: .venv  ## Run pipeline with debug artifacts saved
	$(PYTHON) -m src.main --input "$(CU_FILE)" --debug

watch: .venv  ## Daemon mode: auto‑process new .cu files
	$(PYTHON) -m src.main --daemon

# ── NVIDIA sample fetch ----------------------------------------

fetch-sample:  ## Download NVIDIA/cuda‑samples kernel and run pipeline
	$(PYTHON) -m src.main --nvidia-sample

# ── Cleanup ----------------------------------------------------

clean:  ## Remove build artifacts and proof files
	rm -f $(BINARY) $(PROOF) /tmp/*_proof
	rm -rf __pycache__ .pytest_cache debug/
	rm -f portability_report.json demo_report.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	@echo "  ✅ Clean"

clean-all: clean  ## Clean + remove .venv and ported kernels
	rm -rf .venv
	rm -f ported_kernels/*.hip.cpp
	@echo "  ✅ Full clean"

# ── Help -------------------------------------------------------

help:  ## Show this help menu
	@printf '\n\033[1mKernel Olympics — CUDA → ROCm Migration\033[0m\n\n'
	@printf '  \033[2mUsage:\033[0m    make <target>   [CU_FILE=path/to/kernel.cu]\n\n'
	@printf '  \033[1mCore workflow\033[0m\n'
	@printf '    \033[1;34m%-16s\033[0m %s\n' "pipeline" "Port → compile → run (full cycle)"
	@printf '    \033[1;34m%-16s\033[0m %s\n' "port"    "Run LLM pipeline (CUDA→HIP)"
	@printf '    \033[1;34m%-16s\033[0m %s\n' "compile" "hipcc device-only proof harness"
	@printf '    \033[1;34m%-16s\033[0m %s\n' "run"     "Execute proof on AMD GPU"
	@printf '\n  \033[1mDemo & recording\033[0m\n'
	@printf '    \033[1;34m%-16s\033[0m %s\n' "demo"    "Full migration demo (3-5 min)"
	@printf '    \033[1;34m%-16s\033[0m %s\n' "record"  "asciinema recording"
	@printf '\n  \033[1mQuality\033[0m\n'
	@printf '    \033[1;34m%-16s\033[0m %s\n' "test"    "665 unit tests"
	@printf '    \033[1;34m%-16s\033[0m %s\n' "doctor"  "Pre-flight check"
	@printf '    \033[1;34m%-16s\033[0m %s\n' "lint"    "Ruff linter"
	@printf '\n  \033[1mCache\033[0m\n'
	@printf '    \033[1;34m%-16s\033[0m %s\n' "speed-demo"   "LLM vs cache speed demo"
	@printf '    \033[1;34m%-16s\033[0m %s\n' "fresh"        "Run with fresh cache"
	@printf '\n  \033[1mUtility\033[0m\n'
	@printf '    \033[1;34m%-16s\033[0m %s\n' "clean"        "Remove artifacts"
	@printf '    \033[1;34m%-16s\033[0m %s\n' "clean-all"    "+ remove .venv + ported kernels"
	@printf '    \033[1;34m%-16s\033[0m %s\n' "setup"        "Full first-time setup on new machine"
	@printf '    \033[1;34m%-16s\033[0m %s\n' "reclone"      "Fresh clone (zaps local changes)"
	@printf '\n  \033[1mGitHub sync\033[0m\n'
	@printf '    \033[1;34m%-16s\033[0m %s\n' "pull"         "git pull latest from GitHub"
	@printf '    \033[1;34m%-16s\033[0m %s\n' "push"         "git add+commit+push (MSG=...)"
	@printf '    \033[1;34m%-16s\033[0m %s\n' "sync"         "pull then push (two-way sync)"
	@printf '    \033[1;34m%-16s\033[0m %s\n' "push-to-cation" "Pull here, pull on AMD workspace"
	@printf '\n  \033[1mWatch & debug\033[0m\n'
	@printf '    \033[1;34m%-16s\033[0m %s\n' "debug"   "Debug artifacts saved"
	@printf '    \033[1;34m%-16s\033[0m %s\n' "watch"   "Daemon mode (auto-process)"
	@printf '\n  \033[2mExamples:\033[0m\n'
	@printf '    make pipeline                       # Full cycle (default kernel)\n'
	@printf '    make pipeline CU_FILE=path/to/x.cu  # Full cycle (any kernel)\n'
	@printf '    make demo                           # Recordable demo\n'
	@printf '    make test                           # 665 unit tests\n'
	@printf '    make port-all                       # Port every sample\n'
	@printf '    make pull                           # git pull from GitHub\n'
	@printf '    make push MSG="my fix"              # add + commit + push\n'
	@printf '    make setup                          # Clone + install (new machine)\n'
	@printf '    make reclone                        # Fresh clone from scratch\n'
	@printf '    make push-to-cation                 # Sync to AMD workspace\n'
	@printf '    make debug CU_FILE=path/to/x.cu     # Debug artifacts only\n\n'
