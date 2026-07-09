"""CUDA Kernel Signature Parser — auto-generates spec JSON from CUDA source.

TRIZ #3 (Local Quality): Treats each kernel uniquely by parsing its actual
signature instead of guessing a generic (float*, float*, int) harness.
TRIZ #24 (Intermediary): The spec file mediates between CUDA source and
the HIP compilation harness — bridges the type/signature gap.

Loop engineering: parser → spec → harness → compile → [fail → LLM re-parse] → pass.
The loop converges when the spec matches the kernel's real signature.
"""

import re
import json
from pathlib import Path
from typing import Optional

_SPEC_DIR = Path(__file__).parent / "specs"

# CUDA → HIP type mapping for spec generation
_CUDA_TO_HIP_TYPES = {
    "float": "float",
    "double": "double",
    "int": "int",
    "unsigned int": "unsigned int",
    "long": "long",
    "unsigned long": "unsigned long",
    "long long": "long long",
    "unsigned long long": "unsigned long long",
    "short": "short",
    "unsigned short": "unsigned short",
    "char": "char",
    "unsigned char": "unsigned char",
    "bool": "bool",
    "size_t": "size_t",
    "void": "void",
    "dim3": "dim3",
    "cudaStream_t": "hipStream_t",
    "cudaError_t": "hipError_t",
}

# CUDA pointer types that need direction inference
_POINTER_KEYWORDS = {"input", "in", "src", "d_input", "d_src", "data",
                     "image", "img", "a", "x", "in1", "in2"}
_OUTPUT_KEYWORDS = {"output", "out", "dst", "d_output", "d_dst", "result",
                    "res", "b", "y", "out1", "out2"}


def _infer_direction(param_name: str, param_type: str, index: int) -> str:
    """Infer parameter direction from name conventions + const qualifiers.

    const pointers are always input. Non-const pointer names are heuristic.
    TRIZ #28 (Mechanical Substitution): Use deterministic rules, not LLM.
    """
    if "*" not in param_type and "]" not in param_type:
        return "scalar"
    # const pointer → definitely input (can't write through const)
    if "const" in param_type:
        return "in"
    name_lower = param_name.lower().replace("*", "")
    if len(name_lower) <= 1:
        return "out"  # non-const single-char pointer is likely output
    if name_lower in _OUTPUT_KEYWORDS:
        return "out"
    return "in"


def _guess_grid_block(portable: bool = True) -> dict:
    """Guess a generic launch configuration.

    TRIZ #1 (Segmentation): Separate the signature inference (deterministic)
    from launch config guessing (heuristic). Judges can tune grid/block
    independently of params.
    """
    return {"grid": {"x": 4, "y": 1, "z": 1},
            "block": {"x": 64, "y": 1, "z": 1}}


def _size_expr_from_type(param_type: str, param_name: str) -> str:
    """Generate a reasonable size_expr for a pointer parameter."""
    ptype = param_type.replace("*", "").replace("const ", "").strip()
    if ptype in ("char", "unsigned char"):
        return f"{param_name}_count"
    return "count"


def parse_kernel_signatures(source: str) -> list[dict]:
    """Parse all __global__ function signatures from CUDA source.

    Returns a list of param dicts (one per __global__ function):
      {
        "kernel_function": "shfl_scan_test",
        "params": [{"name": "data", "type": "int*", "direction": "in", "size_expr": "count"}, ...],
        "raw": "int *data, int width, int *partial_sums"
      }

    Uses multiline regex to match __global__ void func_name(...).
    This is deterministic (TRIZ #28 — Mechanical Substitution of LLM guesswork).

    TRIZ #13 (Do It In Reverse): Instead of waiting for compile failure to
    reveal the signature, INVERT — parse the source FIRST, build the spec
    BEFORE the harness. The harness is always correct on the first try.
    """
    # Remove block comments and preprocessor directives for cleaner parsing
    clean = re.sub(r'/\*.*?\*/', '', source, flags=re.DOTALL)
    clean = re.sub(r'//.*', '', clean)
    clean = re.sub(r'#.*', '', clean)

    # Match __global__ void func_name(param1, param2, ...)
    pattern = r'__global__\s+void\s+(\w+)\s*\(([^)]*)\)'
    matches = re.findall(pattern, clean)

    results = []
    for func_name, params_str in matches:
        params_str = params_str.strip()
        if not params_str:
            continue
        # Split by comma, respecting nested parentheses (default values)
        param_list = _split_params(params_str)

        parsed_params = []
        for i, p in enumerate(param_list):
            p = p.strip()
            if not p:
                continue
            # Remove default value: "int *partial_sums = NULL" → "int *partial_sums"
            p_no_default = re.sub(r'\s*=\s*[^,]+$', '', p).strip()
            # Split type and name
            # Types like "const float*" need careful handling
            parts = p_no_default.split()
            if len(parts) == 0:
                continue
            elif len(parts) == 1:
                # Could be a type-only declaration (rare)
                param_type, param_name = parts[0], f"arg{i}"
            else:
                # Last part is the name, preceding parts are the type
                param_name = parts[-1].replace("*", "").lstrip("&")
                param_type = " ".join(parts[:-1])
                # Handle pointers: "float *input" → type="float*", name="input"
                if parts[-1].startswith("*"):
                    param_name = parts[-1].lstrip("*")
                    param_type += "*"

            # Map CUDA type to HIP-equivalent type label for spec
            base_type = param_type.replace("*", "").replace("const ", "").strip()
            hip_label = _CUDA_TO_HIP_TYPES.get(base_type, base_type)

            parsed_params.append({
                "name": param_name,
                "type": param_type,
                "direction": _infer_direction(param_name, param_type, i),
                "size_expr": _size_expr_from_type(param_type, param_name),
            })

        results.append({
            "kernel_function": func_name,
            "params": parsed_params,
            "raw": params_str,
        })

    return results


def _split_params(params_str: str) -> list[str]:
    """Split parameter string respecting nested parentheses.

    int *data, int width, int *partial_sums = NULL
    → ['int *data', 'int width', 'int *partial_sums = NULL']
    """
    parts = []
    depth = 0
    current = []
    for ch in params_str:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current).strip())
    return [p for p in parts if p]


def generate_spec_from_source(kernel_name: str, source: str) -> Optional[dict]:
    """Generate a complete spec dict from CUDA source.

    Returns None if no __global__ function found.
    """
    signatures = parse_kernel_signatures(source)
    if not signatures:
        return None

    # Use the first __global__ function as the primary kernel
    primary = signatures[0]

    spec = {
        "kernel_name": kernel_name,
        "kernel_function": primary["kernel_function"],
        "description": f"Auto-generated spec for {kernel_name}",
        "params": primary["params"],
    }

    # Generate launch config
    spec["launch"] = _guess_grid_block()

    # Count pointer params to guess input setup size
    pointer_count = sum(1 for p in primary["params"]
                        if "*" in p["type"] or "[" in p["type"])
    spec["input_setup"] = {
        "count": 256 * max(1, pointer_count),
        "default_value": 1.0,
    }

    spec["output_readback"] = {
        "count": 4,
        "element_type": "float",
        "format": "float_per_line",
    }

    # If the kernel source has int main(...), mark as self-contained
    if re.search(r'int\s+main\s*\(', source):
        spec["self_contained"] = True

    return spec


def save_spec(kernel_name: str, spec: dict) -> Path:
    """Save a spec dict to the specs/ directory."""
    _SPEC_DIR.mkdir(parents=True, exist_ok=True)
    path = _SPEC_DIR / f"{kernel_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    return path


def auto_generate_spec(kernel_name: str, source: str) -> Optional[dict]:
    """Full pipeline: parse → generate → save → return spec.

    The loop convergence point: if compile fails with harness errors,
    call this to generate a spec, then re-port with the correct harness.
    """
    spec = generate_spec_from_source(kernel_name, source)
    if spec is None:
        return None
    save_spec(kernel_name, spec)
    return spec
