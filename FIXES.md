# Fixes

**T3.1 — Unicode/encoding crash**

- **Centralised the fix.** One module, `src/utf8_console.py`, guards every print site and file I/O call, so all the scattered encoding bugs are caught in a single place instead of patched one by one.
- **Two-tier safety net.** Tier 1 forces stdout/stderr to UTF-8 (what you get on any normal terminal); Tier 2 falls back to readable ASCII transliteration only if a stream genuinely can't do UTF-8 — never a crash, never `?` soup.
- **Example.** On a broken locale, `╔═ Kernel Olympics ●🔍 ═╗` used to raise `UnicodeEncodeError` and kill the run; it now renders as-is under UTF-8, or degrades to `+= Kernel Olympics *[scan] =+` in the fallback.
