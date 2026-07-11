#!/usr/bin/env bash
# Firecrawl Auto-Compile Agent
# Monitors pipeline output for "saved for manual hipcc" markers
# and auto-compiles with device-only harness + runs proof.
#
# Deploy on AMD workspace:
#   bash scripts/firecrawl_auto_compile.sh --setup
#
# Manual run:
#   bash scripts/firecrawl_auto_compile.sh
#
# Cron (every 5 min):
#   */5 * * * * /workspace/Kernel-Olympics/scripts/firecrawl_auto_compile.sh --quiet >> /tmp/firecrawl.log 2>&1

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/Kernel-Olympics}"
PORTED_DIR="${PORTED_DIR:-$REPO_DIR/ported_kernels}"
LOG_FILE="${LOG_FILE:-/tmp/firecrawl_compile.log}"
QUIET="${QUIET:-0}"
MARKER="saved for manual hipcc"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() {
    local level=$1; shift
    echo "[$(date '+%H:%M:%S')] [${level}] $*" | tee -a "$LOG_FILE"
}

info()  { log "${CYAN}INFO${NC}" "$@"; }
ok()    { log "${GREEN}OK${NC}"   "$@"; }
warn()  { log "${YELLOW}WARN${NC}" "$@"; }
err()   { log "${RED}ERR${NC}"    "$@"; }

# ── Setup mode ────────────────────────────────────────────────
setup_mode() {
    cat << 'CRON'
# ── Firecrawl Auto-Compile Agent ──────────────────────────
# Run every 5 minutes. Uses flock to prevent overlapping runs.
# To install:
#   crontab -e
#   Paste the line below (or run the --install-cron option)
#
# CRON LINE (every 5 min, quiet mode):
# */5 * * * * /usr/bin/flock -n /tmp/firecrawl.lock /workspace/Kernel-Olympics/scripts/firecrawl_auto_compile.sh --quiet >> /tmp/firecrawl.log 2>&1
#
# Logs: tail -f /tmp/firecrawl.log
# Run once: bash scripts/firecrawl_auto_compile.sh
CRON
}

# ── Find kernels needing manual compile ──────────────────────
find_manual_kernels() {
    local found=()
    for f in "$PORTED_DIR"/*.hip.cpp; do
        [[ -f "$f" ]] || continue
        base=$(basename "$f" .hip.cpp)
        # Check if it was flagged for manual compile
        if grep -q "$MARKER" "$f" 2>/dev/null; then
            found+=("$f")
        fi
    done
    echo "${found[@]}"
}

# ── Generate device-only proof harness ───────────────────────
generate_proof_harness() {
    local kernel_source="$1"
    local output="$2"
    local width="${3:-32}"  # default 32 (RDNA3)

    python3 -c "
import re, sys

with open('$kernel_source') as f:
    src = f.read()

# Extract all __global__ function signatures and bodies
kernels = re.findall(r'(__global__\s+void\s+(\w+)\s*\([^)]*\)\s*\{)', src)

if not kernels:
    print('No __global__ kernel found', file=sys.stderr)
    sys.exit(1)

# Use the first kernel found
kern_match = re.search(r'__global__\s+void\s+(\w+)\s*\(([^)]*)\)', src)
if not kern_match:
    print('Cannot parse kernel signature', file=sys.stderr)
    sys.exit(1)

kern_name = kern_match.group(1)
full_sig = kern_match.group(0)
params_str = kern_match.group(2)

# Extract scalar parameters to pass from host
params = []
for p in params_str.split(','):
    p = p.strip()
    if p:
        # Skip pointer params (they get d_data)
        if '*' not in p:
            param_name = p.split()[-1] if p.split() else ''
            if param_name and param_name not in ('data', 'd_data') and '=' not in param_name:
                params.append(param_name)

param_defaults = {}
# Parse default values from the kernel signature (e.g., 'int *partial_sums = NULL')
for p in params_str.split(','):
    p = p.strip()
    if '=' in p:
        parts = p.split('=')
        pname = parts[0].strip().split()[-1]
        pval = parts[1].strip()
        param_defaults[pname] = pval

# Determine width to use
w = '$width'
# Check if there's a 'width' param with a default
if 'width' in param_defaults:
    w = param_defaults['width']

# Generate harness
harness = '''#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string.h>

'''

# Extract kernel body only (remove host code)
body_lines = []
in_kernel = False
brace_depth = 0
for line in src.split('\\n'):
    stripped = line.strip()
    if '__global__' in stripped:
        in_kernel = True
        body_lines.append(line)
        # Count initial braces in the signature line
        brace_depth += stripped.count('{') - stripped.count('}')
        continue
    if in_kernel:
        body_lines.append(line)
        brace_depth += line.count('{') - line.count('}')
        if brace_depth <= 0 and line.rstrip().endswith('}'):
            in_kernel = False

harness += '\\n'.join(body_lines)
harness += '''

int main() {
    int n = 256;
    int *d_data, *h_data;
    h_data = (int*)malloc(n * sizeof(int));
    if (!h_data) { fprintf(stderr, \"malloc failed\\n\"); return 1; }
    for (int i = 0; i < n; i++) h_data[i] = i + 1;
    if (hipMalloc(&d_data, n * sizeof(int)) != hipSuccess) {
        fprintf(stderr, \"hipMalloc failed\\n\"); return 1;
    }
    if (hipMemcpy(d_data, h_data, n * sizeof(int), hipMemcpyHostToDevice) != hipSuccess) {
        fprintf(stderr, \"hipMemcpy H2D failed\\n\"); return 1;
    }
    
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
        fprintf(stderr, \"hipGetDeviceProperties failed\\n\"); return 1;
    }
    int width = prop.warpSize;
    printf(\"Device: %s | warpSize=%d\\n\", prop.name, width);
    
    %s<<<1, n>>>(d_data, width, nullptr);
    if (hipDeviceSynchronize() != hipSuccess) {
        fprintf(stderr, \"hipDeviceSynchronize failed\\n\"); return 1;
    }
    
    if (hipMemcpy(h_data, d_data, n * sizeof(int), hipMemcpyDeviceToHost) != hipSuccess) {
        fprintf(stderr, \"hipMemcpy D2H failed\\n\"); return 1;
    }
    
    int pass = 1;
    for (int i = 0; i < n; i++) {
        int base = (i / width) * width;
        int expected = 0;
        for (int k = base; k <= i; k++) expected += (k + 1);
        if (h_data[i] != expected) {
            pass = 0;
            printf(\"FAIL[%d]: got %d, expected %d\\n\", i, h_data[i], expected);
            break;
        }
    }
    printf(\"%s\\n\", pass ? \"PASSED\" : \"FAILED\");
    hipFree(d_data); free(h_data);
    return pass ? 0 : 1;
}
''' % kern_name

with open('$output', 'w') as f:
    f.write(harness)
print('Wrote: $output')
"
}

# ── Main ──────────────────────────────────────────────────────
main() {
    local manual_kernels
    manual_kernels=$(find_manual_kernels)
    
    if [[ -z "$manual_kernels" ]]; then
        [[ "$QUIET" -eq 0 ]] && info "No kernels flagged for manual compile (checking for \"$MARKER\")"
        return 0
    fi
    
    local compiled_any=0
    for hip_file in $manual_kernels; do
        local base
        base=$(basename "$hip_file" .hip.cpp)
        local proof_file="/tmp/${base}_proof.hip.cpp"
        local binary="/tmp/${base}_proof"
        
        info "Found: ${hip_file} — generating proof harness..."
        
        # Detect warpSize from device
        local width=32
        if command -v /tmp/check_warp >/dev/null 2>&1; then
            width=$(/tmp/check_warp 2>/dev/null || echo 32)
        else
            # Create a tiny program to query warpSize
            cat > /tmp/check_warp.hip.cpp << 'EOF'
#include <hip/hip_runtime.h>
#include <cstdio>
int main() {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("%d\n", prop.warpSize);
    return 0;
}
EOF
            hipcc -o /tmp/check_warp /tmp/check_warp.hip.cpp -I/opt/rocm/include 2>/dev/null && \
                width=$(/tmp/check_warp 2>/dev/null || echo 32) || width=32
        fi
        
        info "Detected warpSize=${width}"
        
        # Generate proof harness
        if ! generate_proof_harness "$hip_file" "$proof_file" "$width"; then
            err "Failed to generate harness for ${base}"
            continue
        fi
        
        # Compile
        info "Compiling ${base} with hipcc..."
        compile_out=$(hipcc -o "$binary" "$proof_file" -I/opt/rocm/include 2>&1) || {
            warn "Compile FAILED for ${base}:"
            echo "$compile_out" | tail -5 >> "$LOG_FILE"
            continue
        }
        ok "Compiled ${base} successfully"
        
        # Run
        info "Running ${base}..."
        run_out=$("$binary" 2>&1) || {
            warn "Run FAILED for ${base}:"
            echo "$run_out" >> "$LOG_FILE"
            continue
        }
        echo "$run_out"
        
        if echo "$run_out" | grep -q "PASSED"; then
            ok "✓ ${base}: PASSED on AMD GPU!"
            # Remove the marker from the source file so we don't re-compile
            sed -i '/saved for manual hipcc/d' "$hip_file"
            # Add a proof header
            sed -i '1i // ✓ AUTO-COMPILED and PROVEN on real AMD GPU via firecrawl_auto_compile.sh' "$hip_file"
        else
            warn "✗ ${base}: FAILED — output did not match expected"
            echo "$run_out"
        fi
        compiled_any=1
    done
    
    if [[ "$compiled_any" -eq 1 ]]; then
        ok "Firecrawl cycle complete — $compiled_any kernel(s) compiled"
    fi
}

# ── Parse args ────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --setup|--help)
            setup_mode
            exit 0
            ;;
        --quiet)
            QUIET=1
            shift
            ;;
        --log)
            LOG_FILE="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 [--quiet] [--log FILE] [--setup]"
            exit 1
            ;;
    esac
done

mkdir -p "$(dirname "$LOG_FILE")"
main
