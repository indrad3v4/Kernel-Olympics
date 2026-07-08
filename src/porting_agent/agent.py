"""
Porting Agent — uses Fireworks API to fix CUDA→ROCm porting issues.

Input: flagged kernel + surrounding context + any retrieved similar pattern
Model: Fireworks API (AMD-hosted catalog)
Output: ported code + confidence score + explanation of the fix

Confidence-gated: if confidence < threshold, flag for human review.
"""

import os
import json
from typing import Dict, Optional
from pathlib import Path


class PortingAgent:
    """LLM-based CUDA→ROCm porting agent using Fireworks API."""

    SYSTEM_PROMPT = """You are an expert CUDA→ROCm/HIP migration engineer. 
Your job is to port CUDA kernels to AMD ROCm/HIP, specifically fixing 
warp(32)→wavefront(64) divergence issues.

KEY RULES:
1. AMD GPUs use wavefronts of 64 threads, not warps of 32
2. __shfl_down_sync(0xffffffff, val, 16) on wavefront64 skips half the lanes — 
   the offset must be adjusted or use a different algorithm
3. Hardcoded "32" for warp size → should be "64" or use warpSize/wavefront size
4. __shared__ arrays sized to 32 → may need 64 for wavefront-aware code
5. __syncwarp() → use __syncthreads() for HIP compatibility
6. Use __ballot_sync (HIP) instead of CUDA warp-vote functions
7. Keep the same algorithm structure — only change what's needed for portability
8. __activemask() → use __ballot_sync(0xffffffff, 1) for active lane mask on HIP
9. __all_sync/__any_sync — these take a mask argument; verify it works with 64 lanes
10. __match_all_sync — no direct HIP equivalent; redesign as sequential check
11. threadIdx.x >> 5 computes warp index (32 lanes) → should be >> 6 for wavefront64
12. Lane identification: if (lane_id < 32) → if (lane_id < 64) for wavefront boundary
13. __shfl_sync (basic shuffle) — mask and lane count must be adjusted for wavefront64

Output format: JSON with:
- "ported_code": the full ported kernel
- "confidence": 0-100 score
- "changes": list of specific changes made
- "explanation": short explanation of the fix
"""

    # ✅ VERIFIED WORKING on Fireworks API (tested, confirmed):
    #   - kimi-k2p7-code (strongest: code generation, struct-aware HIP porting)
    #   - glm-5p2        (accurate code generation, struct understanding)
    #   - deepseek-v4-pro (good general fallback)
    # ❌ UNVERIFIED / REMOVED:
    #   - llama-v3p3-70b-instruct  (unstable results on Fireworks, removed)
    FALLBACK_MODELS = [
        "accounts/fireworks/models/kimi-k2p7-code",              # 1st: Kimi K2.7 Code (coder ✅)
        "accounts/fireworks/models/glm-5p2",                      # 2nd: GLM (planner ✅)
        "accounts/fireworks/models/deepseek-v4-pro",              # 3rd: DeepSeek (works ✅)
    ]

    def __init__(self, api_key: Optional[str] = None, model: str = "accounts/fireworks/models/kimi-k2p7-code",
                 deepseek_key: str = "", deepseek_model: str = "deepseek-reasoner"):
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY", "")
        self.model = model
        self.deepseek_key = deepseek_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.deepseek_model = deepseek_model
        self.api_base = "https://api.fireworks.ai/inference/v1"
        self.deepseek_base = "https://api.deepseek.com/v1"

    def _fireworks_api_available(self) -> bool:
        """Quick health check to see if Fireworks API is reachable.

        Uses a short HEAD request with 3s timeout. Returns False when the
        API is unreachable, DNS fails, connection hangs, or any network error
        occurs — so the caller can skip directly to template fallback instead
        of waiting for per-model 5s timeouts.
        """
        try:
            import urllib.request
            req = urllib.request.Request(self.api_base, method="HEAD")
            req.add_header("Authorization", f"Bearer {self.api_key}")
            with urllib.request.urlopen(req, timeout=3):
                return True
        except Exception:
            return False

    def port_kernel(self, source_code: str, context: str = "",
                    cached_pattern: Optional[Dict] = None) -> Dict:
        """Port a CUDA kernel to ROCm/HIP using LLM."""
        
        # TRIZ: Fix source code BEFORE any LLM/template processing
        fixed_source = self._fix_ported_code(source_code)
        
        # Build prompt with context
        user_prompt = f"Port this CUDA kernel to AMD ROCm/HIP:\n\n```cuda\n{fixed_source}\n```\n"
        
        if context:
            user_prompt += f"\nAdditional context:\n{context}\n"
        
        if cached_pattern:
            user_prompt += (
                f"\nA similar pattern was found in memory (confidence: {cached_pattern.get('confidence', 0)}):\n"
                f"Original: {cached_pattern.get('original_snippet', '')}\n"
                f"Verified fix: {cached_pattern.get('verified_fix', '')}\n"
                f"Apply similar approach if applicable.\n"
            )
        
        user_prompt += "\nRespond ONLY with valid JSON matching the expected format."
        
        # For hackathon: if no API key, use template-based porting
        if not self.api_key or self.api_key == "test":
            return self._template_port(fixed_source, cached_pattern)

        # ⏱️ Early health check — skip to template if Fireworks is unreachable
        if not self._fireworks_api_available():
            print("║ ⏱️ Fireworks API unreachable — using template fallback")
            result = self._template_port(source_code, cached_pattern)
            if "ported_code" in result:
                result["ported_code"] = self._fix_ported_code(result["ported_code"])
            return result

        models_to_try = [self.model] + [m for m in self.FALLBACK_MODELS if m != self.model]

        for model in models_to_try:
            try:
                import urllib.request
                import json as _json
                data = _json.dumps({
                    "model": model,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2048,
                    "response_format": {"type": "json_object"}
                }).encode()
                req = urllib.request.Request(
                    f"{self.api_base}/chat/completions",
                    data=data,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    result = _json.loads(resp.read())
                content = result["choices"][0]["message"]["content"]
                try:
                    parsed = _json.loads(content)
                    if "ported_code" in parsed:
                        parsed["ported_code"] = self._fix_ported_code(parsed["ported_code"])
                    return parsed
                except _json.JSONDecodeError:
                    # LLM returned text, not JSON — extract code, use template fallback
                    code_text = self._extract_code_from_text(content)
                    if code_text:
                        return {
                            "ported_code": self._fix_ported_code(code_text),
                            "confidence": self._rubric_score_extracted(code_text),
                            "changes": ["LLM returned text — extracted code block"],
                            "explanation": "Code extracted from LLM text output"
                        }
                    raise
            except Exception as e:
                print(f"║ ⏱️ Model {model} failed in <5s: {e}")
                continue  # Try next model

        # All Fireworks models timed out → skip DeepSeek, go straight to template
        print("║ ⏱️ All Fireworks models timed out — using template fallback")
        result = self._template_port(source_code, cached_pattern)
        if "ported_code" in result:
            result["ported_code"] = self._fix_ported_code(result["ported_code"])
        return result

    @staticmethod
    def _extract_code_from_text(text: str) -> str:
        """Extract HIP/CUDA code from LLM text output (non-JSON responses)."""
        import re
        # Try markdown code blocks with any language
        blocks = re.findall(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
        if blocks:
            # Pick the longest block (usually the real code)
            return max(blocks, key=len).strip()
        # Try to find __global__ kernel definition
        match = re.search(r'(__global__\s+void\s+\w+\s*\(.*?)(?=\n\n|\Z)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Try to find from #include to end
        match = re.search(r'(#include\s+<.*)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def _rubric_score(source_code: str, ported_code: str, changes: list,
                      has_header: bool = False, cached: bool = False) -> int:
        """Rubric-based confidence scoring for ported kernels (0-100).

        Rubric dimensions:
          - Wavefront Header (0-20) : #define WAVEFRONT_SIZE present in output
          - Portability (0-30)      : code is already AMD-ready OR fixes applied
          - Code Integrity (0-30)   : kernel structure preserved, non-empty
          - Change Logging (0-20)   : changes list depth and variety

        Special cases:
          - "No changes needed" (code already AMD-compatible) → full Portability score
          - Cached verified fix applied → high baseline + rubric sanity check
        """
        import re

        # ── Dimension 1: Wavefront Header (0-20) ──
        header_score = 0
        if "#define WAVEFRONT_SIZE 64" in ported_code:
            header_score = 20
        elif "WAVEFRONT_SIZE" in ported_code:
            header_score = 12

        # ── Dimension 2: Portability (0-35) ──
        no_changes_needed = any("no automatic changes needed" in c.lower()
                                for c in changes)
        fix_score = 0
        if no_changes_needed:
            # Code is already AMD-ready — high confidence
            fix_score = 35
        else:
            change_categories = set()
            for c in changes:
                cl = c.lower()
                if "wavefront" in cl:
                    change_categories.add("wavefront")
                if "mask" in cl or "0x3f" in cl or "0x1f" in cl:
                    change_categories.add("mask")
                if "shfl" in cl or "shuffle" in cl:
                    change_categories.add("shuffle")
                if "sync" in cl:
                    change_categories.add("sync")
                if "tile" in cl:
                    change_categories.add("tile")
                if "shared" in cl:
                    change_categories.add("shared_mem")
                if "lane" in cl:
                    change_categories.add("lane_id")
                if "ballot" in cl or "activemask" in cl:
                    change_categories.add("ballot")
                if "all_sync" in cl or "any_sync" in cl or "match_all" in cl:
                    change_categories.add("predicate_sync")
                if "warp_size" in cl or "warp_mask" in cl:
                    change_categories.add("warp_size")
                if "cached" in cl or "verified" in cl:
                    change_categories.add("cached")
            # Baseline (processing was done) + category bonus
            fix_score = min(15 + len(change_categories) * 5, 35)

        # ── Dimension 3: Code Integrity (0-30) ──
        integrity_score = 0
        code_len = len(ported_code.strip())
        if code_len > 0:
            integrity_score += 5
        if code_len > 50:
            integrity_score += 5
        if code_len > 200:
            integrity_score += 5
        if "__global__" in ported_code or "__device__" in ported_code:
            integrity_score += 10
        elif "void" in ported_code and ("(" in ported_code and ")" in ported_code):
            integrity_score += 5
        if source_code and code_len >= len(source_code.strip()) * 0.5:
            integrity_score += 5

        # ── Dimension 4: Change Logging (0-20) ──
        explain_score = 0
        if changes:
            if len(changes) >= 1:
                explain_score += 5
            if len(changes) >= 3:
                explain_score += 5
            if len(changes) >= 5:
                explain_score += 5
            if len(changes) >= 8:
                explain_score += 5

        total = header_score + fix_score + integrity_score + explain_score

        # Cached pattern bonus: +5 if a verified pattern was applied
        if cached:
            total += 5

        return min(total, 100)

    @staticmethod
    def _rubric_score_extracted(code_text: str) -> int:
        """Rubric for code extracted from LLM text output (0-100).

        Lower confidence because extraction is inherently lossy.
        """
        import re
        score = 0

        # Extraction Success (0-25)
        if code_text and len(code_text.strip()) > 0:
            score += 10
        if len(code_text) > 50:
            score += 15

        # Code Completeness (0-40)
        if "__global__" in code_text or "__device__" in code_text:
            score += 25
        elif "void" in code_text and re.search(r'\w+\s*\(', code_text):
            score += 10

        if re.search(r'#include\s*<', code_text):
            score += 10
        if re.search(r'\{[^}]*\}', code_text, re.DOTALL):
            score += 5

        # Structural Validity (0-35)
        if re.search(r'__global__\s+void\s+\w+\s*\(', code_text):
            score += 15
        if re.search(r'threadIdx|blockIdx|blockDim', code_text):
            score += 10
        if re.search(r'(__shared__|__device__|__constant__)', code_text):
            score += 10

        return min(score, 100)

    @staticmethod
    def _fix_ported_code(code: str) -> str:
        """Fix AMD-specific issues in ported code.
        
        Fixes:
        - 32-bit __shfl mask → 64-bit for wavefront64
        """
        import re
        # Count how many 32-bit masks remain
        mask_pattern = re.compile(r'(__shfl_\w+_sync\()0x[fF]{8}(,)')
        before = len(mask_pattern.findall(code))
        code = mask_pattern.sub(r'\g<1>0xffffffffffffffffULL\g<2>', code)
        after = len(mask_pattern.findall(code))
        if before > 0 and after == 0:
            pass  # All masks fixed
        elif before > 0:
            print(f"║ ⚠️ Mask fix: {before} found, {after} remaining (regex issue!)")
        return code

    def _template_port(self, source_code: str,
                       cached_pattern: Optional[Dict] = None) -> Dict:
        """Template-based porting for when API is unavailable (demo fallback)."""

        import re
        changes = []
        lines = source_code.split('\n')
        result_lines = []
        wavefront_header_added = False
        has_added_wave64_shfl = False

        # Template transformations (only on non-comment lines)
        shared_32_re = re.compile(r'(__shared__[^;]*?\[\s*)32(\s*\])')
        tile_32_re = re.compile(r'(tile\[)\s*32(\s*\]\[)\s*32(\s*\])')
        blockidx_32_re = re.compile(r'(blockIdx\.[xy])\s*\*\s*32\s*\+')
        syncwarp_re = re.compile(r'__syncwarp\(\s*\)')
        warp_size_re = re.compile(r'(?:const\s+)?int\s+WARP_SIZE\s*=\s*32')
        warp_size_define_re = re.compile(r'#define\s+WARP_SIZE\s+32\b')
        ballot_re = re.compile(r'__ballot_sync\(0xffffffff')
        shfl_xor_re = re.compile(r'__shfl_xor_sync\s*\(')
        threadidx_32_re = re.compile(r'(threadIdx\.[xy]\s*\*\s*)32(\b)')
        define_tile_re = re.compile(r'#define\s+TILE_SIZE\s+32\b')
        warp_mask_re = re.compile(r'(?:const\s+)?int\s+WARP_MASK\s*=\s*0x1[fF]\b')
        tid_warp_mask_re = re.compile(r'(tid\s*&\s*)0x1[fF](\s*\)?\s*==\s*0\b)')
        blockidx_tile_re = re.compile(r'(blockIdx\.[xy]\s*\*\s*)TILE_SIZE')
        shfl_down_re = re.compile(r'__shfl_down_sync\s*\(')
        activemask_re = re.compile(r'__activemask\s*\(\s*\)')
        all_sync_re = re.compile(r'__all_sync\s*\(')
        any_sync_re = re.compile(r'__any_sync\s*\(')
        match_all_re = re.compile(r'__match_all_sync\s*\(')
        warp_lane_shift_re = re.compile(r'(threadIdx\.[xy]\s*>>\s*)5(?!\d)')
        lane_id_32_re = re.compile(r'(lane_id|laneIdx)\s*[<]\s*32\b')
        warp_divergent_32_re = re.compile(r'(if\s*\(\s*(?:threadIdx\.[xy]|tid|lane_id|laneIdx)\s*[<]\s*)32(\s*\))')

        for line in lines:
            stripped = line.strip()

            # Skip comment-only lines
            if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
                result_lines.append(line)
                continue

            # Track if this line was modified
            original = line

            # Fix 1: Hardcoded 32 in shared memory → change to 64
            line = shared_32_re.sub(r'\1 WAVEFRONT_SIZE \2', line)

            # Fix 2: Hardcoded 32 in tile declarations
            line = tile_32_re.sub(r'\1 WAVEFRONT_SIZE \2 WAVEFRONT_SIZE \3', line)

            # Fix 3: Hardcoded 32 in block indexing
            line = blockidx_32_re.sub(r'\1 * WAVEFRONT_SIZE +', line)

            # Fix 4: __syncwarp() → __syncthreads()
            if syncwarp_re.search(line):
                line = syncwarp_re.sub('__syncthreads();  // wavefront64: full block sync', line)
                if "wavefront64: full block sync" not in __import__('json').dumps(changes):
                    changes.append("__syncwarp() → __syncthreads() for HIP compatibility")

            # Fix 5: __shfl_down_sync — no safe automatic fix for offset semantics
            # (The actual fix depends on algorithm context; LLM handles this best)
            # But we CAN prepend offset=32 for wavefront64 (6 steps → 64 elements)
            
            # Fix 5b: Insert 6th shuffle offset for wavefront64
            if 'shfl_down' in stripped and 'val += __shfl' in line:
                if not has_added_wave64_shfl:
                    has_added_wave64_shfl = True
                    indent = line[:len(line) - len(line.lstrip())]
                    new_line = f"{indent}val += __shfl_down_sync(0xffffffffffffffffULL, val, 32);  // ADDED: wavefront64 offset\n{line}"
                    result_lines[-1] = new_line
                    if "wavefront64_offset32" not in str(changes):
                        changes.append("wavefront64: added offset=32 shuffle step (6-step reduction for 64 lanes)")
                    continue

            # Fix 6: 0x1f (warp mask 32) → 0x3f (wavefront mask 64)
            if '0x1f' in line and not stripped.startswith('//'):
                line = line.replace('0x1f', '0x3f')

            # Fix 7: Hardcoded WARP_SIZE = 32 (with or without const)
            line = warp_size_re.sub('const int WAVEFRONT_SIZE = 64;  // AMD wavefront', line)
            
            # Fix 7b: #define WARP_SIZE 32 (preprocessor macro style)
            if '#define WARP_SIZE' in line and warp_size_define_re.search(line):
                line = warp_size_define_re.sub('#define WAVEFRONT_SIZE 64  // AMD wavefront', line)
                if '#define WAVEFRONT_SIZE 64' not in ' '.join(changes):
                    changes.append("#define WARP_SIZE 32 → #define WAVEFRONT_SIZE 64")

            # Fix 8: __ballot_sync — fix mask and annotate
            if ballot_re.search(line):
                line = ballot_re.sub('__ballot_sync(0xffffffffffffffffULL', line)
                if "ballot_sync mask" not in str(changes):
                    changes.append("__ballot_sync mask → 0xffffffffffffffffULL for wavefront64")
            
            # Fix 8b: __shfl_xor_sync — annotate as wavefront-dependent
            if shfl_xor_re.search(line):
                # Safest auto-fix: add comment; actual offset fix is algorithm-dependent
                if "shfl_xor" not in str(changes):
                    changes.append("__shfl_xor_sync: verify XOR offsets work with wavefront64 (64 lanes, not 32)")
            
            # Fix 8c: threadIdx.* 32 pattern (pointer arithmetic, e.g., &shared[threadIdx.y * 32])
            line = threadidx_32_re.sub(r'\1 WAVEFRONT_SIZE ', line)

            # Fix 8d: #define TILE_SIZE 32 → #define TILE_SIZE WAVEFRONT_SIZE
            if define_tile_re.search(line):
                line = define_tile_re.sub('#define TILE_SIZE WAVEFRONT_SIZE  // AMD wavefront', line)
                if "#define TILE_SIZE WAVEFRONT_SIZE" not in ' '.join(changes):
                    changes.append("#define TILE_SIZE 32 → #define TILE_SIZE WAVEFRONT_SIZE")

            # Fix 8e: int WARP_MASK = 0x1f → int WAVEFRONT_MASK = 0x3f
            if warp_mask_re.search(line):
                line = warp_mask_re.sub('int WAVEFRONT_MASK = 0x3f;  // wavefront64 mask', line)
                if "WARP_MASK → WAVEFRONT_MASK" not in str(changes):
                    changes.append("WARP_MASK 0x1f (32) → WAVEFRONT_MASK 0x3f (64)")

            # Fix 8f: tid & 0x1f == 0 → tid & 0x3f == 0 (warp mask check)
            line = tid_warp_mask_re.sub(r'\1 0x3f\2', line)

            # Fix 8g: blockIdx.* * TILE_SIZE → blockIdx.* * WAVEFRONT_SIZE
            line = blockidx_tile_re.sub(r'\1 WAVEFRONT_SIZE', line)

            # Fix 8h: __shfl_down_sync — annotate the offset issue
            if shfl_down_re.search(line):
                if "shfl_down" not in str(changes):
                    changes.append("__shfl_down_sync: verify offsets work with wavefront64 (64 lanes, offset must be power of two)")
            
            # Fix 8i: 32-bit mask → 64-bit mask for AMD wavefront64
            mask_line = re.sub(r'(__shfl_\w+_sync\()0x[fF]{8}(,)', r'\g<1>0xffffffffffffffffULL\g<2>', line)
            if mask_line != line:
                line = mask_line
                if "mask_64bit" not in str(changes):
                    changes.append("__shfl_*_sync: mask 0xffffffff → 0xffffffffffffffffULL (64-bit for wavefront64)")

            # Fix 9: __activemask() → __ballot_sync(0xffffffffffffffffULL, 1) on HIP
            if activemask_re.search(line):
                line = activemask_re.sub('__ballot_sync(0xffffffffffffffffULL, 1)', line)
                if "activemask" not in str(changes):
                    changes.append("__activemask() → __ballot_sync(0xffffffffffffffffULL, 1) for HIP compatibility")

            # Fix 10: __all_sync / __any_sync — annotate for wavefront64
            if all_sync_re.search(line):
                if "all_sync" not in str(changes):
                    changes.append("__all_sync: verify predicate works with wavefront64 (64 lanes)")
            if any_sync_re.search(line):
                if "any_sync" not in str(changes):
                    changes.append("__any_sync: verify predicate works with wavefront64 (64 lanes)")

            # Fix 11: __match_all_sync — annotate (no direct HIP equivalent)
            if match_all_re.search(line):
                if "match_all" not in str(changes):
                    changes.append("__match_all_sync: no direct HIP equivalent — may need algorithm redesign")

            # Fix 12: threadIdx.x >> 5 (warp index) → >> 6 for wavefront64
            line = warp_lane_shift_re.sub(r'\1 6;  // wavefront64: 64 lanes', line)

            # Fix 13: lane_id < 32 → lane_id < 64 (wavefront boundary)
            if lane_id_32_re.search(line):
                line = line.replace('< 32', '< WAVEFRONT_SIZE', 1)
                if "lane_id < 32" not in str(changes):
                    changes.append("lane_id < 32 → lane_id < WAVEFRONT_SIZE for wavefront64")

            # Fix 14: threadIdx.x/tid < 32 → < WAVEFRONT_SIZE (warp divergence boundary)
            line = warp_divergent_32_re.sub(r'\1 WAVEFRONT_SIZE \2', line)


            # Track what changed
            if line != original:
                # Compute a change description based on what was modified
                if 'WAVEFRONT_SIZE' in line and 'WAVEFRONT_SIZE' not in original:
                    if 'shared' in original and '32' in original:
                        changes.append("__shared__ array sized 32 → WAVEFRONT_SIZE for wavefront64")
                    elif 'blockIdx' in original:
                        changes.append("blockIdx.*32 → blockIdx.*WAVEFRONT_SIZE")
                    elif 'tile' in original:
                        changes.append("tile[32][32] → tile[WAVEFRONT_SIZE][WAVEFRONT_SIZE]")
                    elif 'threadIdx' in original and '* 32' in original:
                        changes.append("threadIdx.*32 → threadIdx.*WAVEFRONT_SIZE in pointer arithmetic")
                if '0x3f' in line and '0x1f' in original:
                    changes.append("Warp mask 0x1f (32) → 0x3f (64) for wavefront64")
                if 'WAVEFRONT_SIZE = 64' in line and 'WARP_SIZE' in original:
                    changes.append("WARP_SIZE = 32 → WAVEFRONT_SIZE = 64")
                if 'TILE_SIZE WAVEFRONT_SIZE' in line and 'TILE_SIZE 32' in original:
                    changes.append("#define TILE_SIZE 32 → #define TILE_SIZE WAVEFRONT_SIZE")
                if 'WAVEFRONT_MASK' in line and 'WARP_MASK' in original:
                    changes.append("WARP_MASK → WAVEFRONT_MASK")
                if '0x3f' in line and '0x1f' in original and '#define' not in original and 'WARP_MASK' not in original:
                    changes.append("tid & 0x1f → tid & 0x3f for wavefront64")

            result_lines.append(line)

        # Fix 9: Add wavefront awareness header (unless already present or first line has it)
        code = '\n'.join(result_lines)
        if "#define WAVEFRONT_SIZE 64" not in code:
            code = "#define WAVEFRONT_SIZE 64  // AMD GPU wavefront size\n" + code
            changes.append("Added #define WAVEFRONT_SIZE 64 header")

        # Deduplicate changes
        seen = set()
        unique_changes = []
        for c in changes:
            if c not in seen:
                seen.add(c)
                unique_changes.append(c)

        # Apply cached pattern if available
        has_cached = cached_pattern is not None
        if has_cached:
            unique_changes.append(f"Applied cached pattern from verified fix (id: {cached_pattern.get('id', 'unknown')})")
            if cached_pattern.get("verified_fix"):
                code = cached_pattern["verified_fix"]

        # Rubric-based confidence scoring
        has_wavefront_header = "#define WAVEFRONT_SIZE 64" in code
        confidence = self._rubric_score(source_code, code, unique_changes,
                                        has_header=has_wavefront_header,
                                        cached=has_cached)

        # When a verified cached fix is used, confidence should reflect
        # the cached pattern's stored verification result (known-good code).
        if has_cached and cached_pattern.get("verified_fix"):
            cached_conf = cached_pattern.get("confidence", 0.85)
            if cached_conf < 1:
                cached_conf = cached_conf * 100
            confidence = max(confidence, int(cached_conf))

        return {
            "ported_code": code,
            "confidence": confidence,
            "changes": unique_changes if unique_changes else ["No automatic changes needed — code appears portable"],
            "explanation": "Template-based porting applied. "
                          f"Made {len(unique_changes)} changes. "
                          "For production, use Fireworks API for better accuracy."
        }
