"""UTF-8 console setup with a readable ASCII fallback.

The live progress display prints box-drawing characters and emoji. On a
terminal whose encoding is not UTF-8 — Windows legacy code pages, a POSIX
``C``/``ASCII`` locale, or output piped under such a locale — every
``print()`` of those glyphs raises ``UnicodeEncodeError`` and crashes the
run.

``enable_utf8_console()`` reconfigures stdout/stderr to UTF-8. If the
stream genuinely cannot be coerced to UTF-8, it installs a transliterating
writer so glyphs degrade to ASCII instead of crashing.
"""
import sys

# Unicode glyph (ordinal) -> ASCII replacement, used only when the stream
# cannot be coerced to UTF-8. Keeps the layout legible instead of turning
# every glyph into '?'. Combined sequences (e.g. U+26A0 U+FE0F) transliterate
# per-codepoint: U+26A0 -> "[warn]", U+FE0F -> "".
_TRANSLATE = {
    0x2550: '=', 0x2500: '-', 0x2502: '|', 0x2551: '|',
    0x2554: '+', 0x2557: '+', 0x255A: '+', 0x255D: '+',
    0x2560: '+', 0x2563: '+', 0x2566: '+', 0x2569: '+', 0x256C: '+',
    0x251C: '+', 0x2524: '+',
    0x25CF: '*', 0x25CB: 'o', 0x2713: 'v', 0x2714: 'v', 0x2717: 'x', 0x2718: 'x',
    0x2192: '->', 0x2190: '<-', 0x2191: '^', 0x2193: 'v',
    0x2014: '--', 0x2013: '-', 0x00D7: 'x', 0x2022: '*',
    0x23E9: '>>', 0x26A0: '[warn]', 0xFE0F: '',
    0x1F50D: '[scan]', 0x1F9E0: '[mem]', 0x1F916: '[port]',
    0x2705: '[ok]', 0x274C: '[x]', 0x1F4CA: '[report]',
    0x1F4C1: '[file]', 0x1F320: '*',
}


def _stream_handles_unicode(stream) -> bool:
    """True if *stream* can encode the display's non-ASCII glyphs."""
    enc = getattr(stream, "encoding", None)
    if not enc:
        return False
    try:
        "═●→".encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


class _AsciiFallbackWriter:
    """Wrap a text stream, transliterating non-ASCII glyphs on write.

    Every other attribute (``isatty``, ``flush``, ``fileno``,
    ``reconfigure``, ...) delegates to the wrapped stream so callers that
    poke at stdout keep working.
    """

    def __init__(self, stream):
        self._stream = stream

    def write(self, s):
        return self._stream.write(s.translate(_TRANSLATE))

    def __getattr__(self, name):
        return getattr(self._stream, name)


def enable_utf8_console() -> None:
    """Make stdout/stderr safe for the progress display.

    Idempotent. Tries UTF-8 first; only falls back to ASCII
    transliteration when the stream genuinely cannot encode UTF-8.
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None or isinstance(stream, _AsciiFallbackWriter):
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass
        if not _stream_handles_unicode(stream):
            setattr(sys, name, _AsciiFallbackWriter(stream))
