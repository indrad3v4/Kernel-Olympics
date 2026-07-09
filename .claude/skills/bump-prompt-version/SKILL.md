---
name: bump-prompt-version
description: Change a system prompt in src/router.py and keep PROMPT_VERSION, prompts/CHANGELOG.md, and data/prompt_changelog.json in sync. Use when editing any string in SYSTEM_PROMPTS, or when asked to "bump the prompt version" or "add a changelog entry".
---

# Bump Prompt Version

Every system prompt in [`src/router.py`](../../../src/router.py) carries a `[prompt vX.Y.Z]`
tag matching `router.PROMPT_VERSION`. Without it there is no way to tell which prompt text
produced a given run. Three artifacts must agree, and `tests/test_demo_budget.py` fails if
they drift:

1. `PROMPT_VERSION` in `src/router.py`
2. `prompts/CHANGELOG.md`
3. `data/prompt_changelog.json` (`current` key **and** an entry in `versions`)

## Bumping rules

| Bump | When |
|---|---|
| MAJOR `v1.0.0 → v2.0.0` | Behavioral change: new instructions, removed constraints |
| MINOR `v1.0.0 → v1.1.0` | Context additions: new examples, expanded edge cases |
| PATCH `v1.0.0 → v1.0.1` | Wording, typos, formatting |

Editing a prompt's *meaning* is MAJOR even if the diff is one word.

## Steps

1. Edit the prompt string in `SYSTEM_PROMPTS`. Leave the `[prompt {PROMPT_VERSION}]` f-string
   interpolation alone — it picks up the new version automatically.
2. Bump `PROMPT_VERSION` in `src/router.py`.
3. Add an entry at the **top** of the version list in `data/prompt_changelog.json` and set
   `current` to the new version:

```json
{ "version": "v1.1.0", "date": "YYYY-MM-DD", "author": "...",
  "changes": ["..."], "files_changed": ["src/router.py"], "run_results": {} }
```

4. Add a matching section to `prompts/CHANGELOG.md`, newest first.
5. Verify the three agree:

```bash
python -m pytest tests/test_demo_budget.py -q -k PromptVersioning
```

6. Run the full suite — several tests match on system-prompt substrings:

```bash
python -m pytest tests/ -x -q
```

## Where the prompts live

All four are in `SYSTEM_PROMPTS`, including `glm_error_analyst`, which was moved out of its
inline call site in `route()` so no prompt escapes versioning. If you add a fifth, it must
carry the tag — `test_every_system_prompt_carries_the_version` enumerates the dict.

The phase prompts (`_build_kimi_refine_prompt`, `_build_deepseek_plan_prompt`, …) are built at
call time and are **not** versioned strings. Changing one is still a prompt change: bump anyway
and say so in the changelog.

## Troubleshooting

- **`test_changelog_json_matches_the_code_version` fails** — `current` in the JSON does not equal
  `PROMPT_VERSION`, or the version is missing from `versions`.
- **`test_stagnation_triggers_replan` fails after a GLM prompt edit** — that test routes on the
  substring `"error analyst"` in the system prompt. Keep the phrase.
- **The changelog file "disappears" on a fresh clone** — check `.gitignore`. `data/` is ignored as
  `data/*` precisely so `!data/prompt_changelog.json` can re-include it; git cannot re-include a
  file whose parent *directory* is excluded.

## Anti-patterns

- **Do not edit a prompt without bumping.** A run's `prompt_version` then points at text that no
  longer exists, which is worse than no versioning: the record is confidently wrong.
- **Do not hardcode a version literal into a prompt string.** Interpolate `{PROMPT_VERSION}`, so
  one edit updates every tag.
- **Do not hardcode tuning constants into prompt text.** The refine prompt used to say
  "stagnation threshold: 3" while the code used a different number. Read the constant
  (`STAGNATION_ABORT_THRESHOLD`) instead.
- **Do not reuse a version for a different prompt.** `run_results` in the JSON is keyed on the
  assumption that one version means one exact prompt text.
