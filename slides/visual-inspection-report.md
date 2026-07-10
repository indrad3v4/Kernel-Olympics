# Visual Inspection Report — Kernel Olympics Slide Deck

**Inspected:** 9 slides (slide-1.jpg through slide-9.jpg)
**Project:** AMD Developer Hackathon ACT II — Track 3, Team Meteorite
**Date:** 2026-07-10

---

## Slide 1 — Title Slide ("KERNEL OLYMPICS")

| Issue | Severity | Detail |
|-------|----------|--------|
| **Low-contrast text** | Medium | The tagline ("Ship AMD-ready code in minutes, not months.") and hackathon info line are in light gray on dark navy — hard to read at projection distance. |
| **Insufficient bottom margin** | Medium | Footer links ("github.com/indrad3v4/Kernel-Olympics" / "endearing-rebirth.up.railway.app") are on a red bar that sits < 0.5" from the slide bottom edge. |
| **Tight spacing above decorative line** | Minor | The thin red rule below the tagline is very close to the tagline itself — needs ~2–3pt more breathing room. |

---

## Slide 2 — Problem Slide ("The $10B Problem")

| Issue | Severity | Detail |
|-------|----------|--------|
| **Low-contrast body text** | Medium | The italicized sentence at the bottom of the right card ("MI300X outperforms NVIDIA on price/performance...") is gray italic on white — nearly invisible from a distance. |
| **Low-contrast footer text** | Medium | Footer tagline ("Enterprises stay on CUDA because...") is white text on dark blue but very small (likely < 14pt), reducing readability. |
| **Header understated** | Minor | "The $10B Problem" title is relatively small for a headline — doesn't anchor the slide visually. |
| **Jargon without context** | Minor | "warp ops, shfl, libraries" — these CUDA-specific terms may not resonate with a general audience without a brief qualifier. |

---

## Slide 3 — Solution Overview (6-step flow)

| Issue | Severity | Detail |
|-------|----------|--------|
| **Low-contrast subtitle** | Minor | The subtitle ("A 4-LLM agentic loop...") is light gray on dark navy — legible but could be brighter. |
| — | — | No structural issues found. Flowchart boxes are well-spaced, text fits, no overlaps. |

---

## Slide 4 — Architecture Detail

| Issue | Severity | Detail |
|-------|----------|--------|
| **Low-contrast title** | Medium | "Architecture: Multi-Agent Orchestration" — white text on light gray/cream background bar is washed out and hard to read. |
| — | — | No layout or spacing issues. Innovation boxes (3-gate validation / pattern memory / strip-to-kernel) are well-proportioned. |

---

## Slide 5 — Smart Porting (before/after code)

| Issue | Severity | Detail |
|-------|----------|--------|
| **Spelling errors in code** | Medium | Three apparent typos visible in the CUDA/HIP code blocks: |
| | | → "**Corporaton**" (should be "Corporation") |
| | | → "**runtim**" (should be "runtime") |
| | | → "**DeVice**" (inconsistent capitalization — may be intentional but looks like an error) |
| — | — | No layout/overflow issues. Code blocks are readable with good contrast. |

> **Note:** If these are deliberately introduced typo-examples to illustrate code problems that Kernel Olympics fixes, consider making that visually clearer (e.g. red strike-through or annotation). Otherwise they look like genuine errors.

---

## Slide 6 — Three-Gate Validation

| Issue | Severity | Detail |
|-------|----------|--------|
| **Low-contrast title** | Medium | "Three-Gate Validation System" — white text on the light gray header bar is washed out, similar to slide 4. |
| **Uneven vertical gaps** | Minor | The gap between the title and the gate cards is noticeably larger than the gap between the gate cards and the footer — creates a slight visual drift. |

---

## Slide 7 — Performance & Speed

| Issue | Severity | Detail |
|-------|----------|--------|
| **Text overflow / cut-off** | **High** | "**Olym pics**" — the word "Olympics" in the table third row is being split awkwardly across two lines due to a narrow cell. Should be on one line. |
| **Low-contrast metric labels** | Medium | The gray description text (e.g. "Pipeline per kernel", "Average cost") in the dark blue metric boxes is hard to read — needs lighter shade. |
| **Inconsistent gaps between metric cards** | Minor | The gap between the first and second metric boxes is slightly larger than between the second and third. |
| **Footer banner too close to bottom edge** | Medium | The red footer banner with the pattern memory stat is flush or nearly flush with the slide bottom margin (< 0.5"). |
| **Term inconsistency** | Minor | "hpify" (lowercase h) appears in the table — elsewhere in the deck it's "hipify". Should be consistent. |

---

## Slide 8 — Team Meteorite

| Issue | Severity | Detail |
|-------|----------|--------|
| **Low-contrast title** | Medium | "Team Meteorite" — white text on the light gray header bar, same washed-out look as slides 4 and 6. |
| **Low-contrast descriptions** | Minor | The gray role-description text within each team card is faint and could be bumped 1-2 shades darker for readability. |
| — | — | Card grid layout is clean and even. No overlap or spacing issues. |

---

## Slide 9 — Thank You

| Issue | Severity | Detail |
|-------|----------|--------|
| **Low-contrast subtitle** | Medium | "Try Kernel Olympics — it's open source" in light gray on dark navy — needs to be white or a much lighter gray. |
| **Footer bar flush with bottom edge** | Medium | The red footer bar spans edge-to-edge with zero side margin and sits at the very bottom of the slide canvas — looks intentionally full-bleed but might clip on projectors with overscan. |
| **Uneven vertical spacing** | Minor | The gap from "THANK YOU" to the subtitle is larger than the gap from the subtitle to the buttons; the gap from "Questions?" to the footer bar is also tight. Inconsistent rhythm. |

---

## Cross-Slide Issues (Systemic)

| # | Issue | Affected Slides |
|---|-------|-----------------|
| 1 | **Low-contrast title text** — white text on light gray/cream header bar is a recurring problem | 4, 6, 8 |
| 2 | **Low-contrast gray body text** on dark navy — need ≥ #CCCCCC or use white | 1, 3, 7, 9 |
| 3 | **Footer/bottom margin < 0.5"** on slides with red bottom bars | 1, 7, 9 |
| 4 | **Inconsistent vertical rhythm** — gaps between elements vary across sections | 6, 9 |
| 5 | **Spelling/consistency errors** in code blocks and table labels | 5, 7 |

---

## Summary

- **Critical issues:** 0
- **High-severity issues:** 1 (text cut-off "Olym pics" on slide 7)
- **Medium-severity issues:** 10 (low contrast across 5 slides, bottom margins on 3 slides, spelling typos on slide 5)
- **Minor issues:** 6 (jargon, tight spacing, inconsistent gaps, term inconsistency)
- **Total issues found: 17**
