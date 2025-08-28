"""
Tiny terminal diagnostics renderer 

*   Rust‑style code‑frame errors (primary/secondary labels, multi‑line spans).
*   Tab‑aware column math, configurable context lines & tab width.
*   Pure Python 3.8+, zero deps, optional ANSI colours.
*   Pluggable `SourceProvider` (filesystem, in‑memory, etc.).

Run the demo at the bottom to preview three different diagnostics:

error: syntax error: expected expression
  --> demo.cool:2:13
   |
 1 | fun main() {
 2 |     let x = 1 + ;
   |             ^     expected expression
   |               ~   after '+'
 3 |     print(x)
  = note help: try putting an identifier or literal
error: redefinition of 'total'
  --> multi.cool:4:9
   |
 1 | fn compute(a, b, c) {
 2 |     let total = a +
   |         ~~~~~       previous definition is here
 3 |         b + c;
 4 |     let total = 0;  // shadowing error
   |         ^^^^^                          'total' redefined here
 5 |     return total
warning: line break after '+' requires an expression
  --> multi.cool:2:20
   |
 1 | fn compute(a, b, c) {
 2 |     let total = a +
   |                   ^ expected expression
 3 |         b + c;
   | ^^^^^^^^      
   |         ~      starts next term
 4 |     let total = 0;  // shadowing error
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, TextIO, Tuple
import sys

Severity = Literal["error", "warning", "note", "help"]

@dataclass(frozen=True)
class Span:
    """1‑based UTF‑8 character span (half‑open)."""

    file: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int

    def normalized(self) -> Span:
        if (self.end_line, self.end_col) < (self.start_line, self.start_col):
            return Span(self.file, self.end_line, self.end_col, self.start_line, self.start_col)

        return self

@dataclass(frozen=True)
class Label:
    span: Span
    message: Optional[str] = None
    kind: Literal["primary", "secondary"] = "primary"

@dataclass
class Diagnostic:
    severity: Severity
    message: str
    labels: List[Label] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

class SourceProvider:
    """Abstract interface."""
    def get(self, _path: str) -> Optional[str]:
        raise NotImplementedError

class FileSystemSourceProvider(SourceProvider):
    def get(self, path: str) -> Optional[str]:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError:
            return None

class MemorySourceProvider(SourceProvider):
    def __init__(self, files: Dict[str, str]):
        self.files = dict(files)

    def get(self, path: str) -> Optional[str]:
        return self.files.get(path)

class TerminalTheme:
    def __init__(self, use_color: Optional[bool] = None):
        if use_color is None:
            use_color = sys.stdout.isatty()
        self.use_color = use_color
        self.palette = {
            "reset": "\x1b[0m",
            "error": "\x1b[31m",
            "warning": "\x1b[33m",
            "note": "\x1b[34m",
            "help": "\x1b[36m",
            "primary": "\x1b[31m",
            "secondary": "\x1b[35m",
            "gutter": "\x1b[90m",
        }

    def c(self, key: str, text: str) -> str:
        if not self.use_color:
            return text

        return f"{self.palette.get(key, '')}{text}{self.palette['reset']}"

@dataclass
class RenderOptions:
    tab_width: int = 4
    context_lines: int = 1
    use_unicode: bool = True
    show_gutter: bool = True

class TerminalRenderer:
    """Pretty‑prints `Diagnostic` objects to a stream."""

    def __init__(self, options: RenderOptions | None = None, theme: TerminalTheme | None = None):
        self.opt = options or RenderOptions()
        self.theme = theme or TerminalTheme()
        self._default_src = FileSystemSourceProvider()

    def render(self, diag: Diagnostic, *, out: TextIO = sys.stdout, source_provider: Optional[SourceProvider] = None) -> None:
        out.write(self.format(diag, source_provider) + "\n")

    def format(self, diag: Diagnostic, source_provider: Optional[SourceProvider] = None) -> str:
        sp = source_provider or self._default_src
        anchor = next((l for l in diag.labels if l.kind == "primary"), diag.labels[0] if diag.labels else None)
        header = self._header(diag.severity, diag.message, anchor)

        body_parts: List[str] = []
        for path, lbls in _group_by_file(diag.labels).items():
            body_parts.append(self._file_chunk(path, lbls, sp.get(path)))
        notes = "\n".join(f"  = {self.theme.c('note', 'note')} {n}" for n in diag.notes) if diag.notes else ""

        return "\n".join(s for s in [header, *body_parts, notes] if s)

    def _header(self, sev: Severity, msg: str, anchor: Optional[Label]) -> str:
        prefix = self.theme.c(sev, sev)
        if anchor is None:
            return f"{prefix}: {msg}"
        s = anchor.span.normalized()

        return f"{prefix}: {msg}\n  --> {s.file}:{s.start_line}:{s.start_col}"

    def _file_chunk(self, path: str, labels: Sequence[Label], text: Optional[str]) -> str:
        gutter_on = self.opt.show_gutter
        g_head = "   |" if gutter_on else ""
        lines: List[str] = [g_head]

        if text is None:
            for lab in labels:
                s = lab.span.normalized()
                kind = self.theme.c(lab.kind, lab.kind)
                msg = f" {lab.message}" if lab.message else ""
                lines.append(f"{g_head} (source unavailable) {path}:{s.start_line}:{s.start_col} {kind}{msg}")
            return "\n".join(lines)

        raw = text.splitlines()
        file_len = len(raw)

        target_lines: set[int] = set()
        for lab in labels:
            s = lab.span.normalized()
            target_lines.update(range(s.start_line, s.end_line + 1))

        context: set[int] = set()
        for ln in target_lines:
            for d in range(-self.opt.context_lines, self.opt.context_lines + 1):
                k = ln + d
                if 1 <= k <= file_len:
                    context.add(k)

        show = sorted(context)
        width = len(str(show[-1])) if show else 1

        cache: Dict[int, Tuple[str, List[int]]] = {}
        def expand(ln: int) -> Tuple[str, List[int]]:
            if ln not in cache:
                expanded, m = _expand_tabs_with_map(raw[ln - 1], self.opt.tab_width)
                cache[ln] = (expanded, m)
            return cache[ln]

        by_line: Dict[int, List[Tuple[Label, int, int, bool]]] = {}
        dangling: List[Label] = []
        for lab in labels:
            s = lab.span.normalized()
            in_bounds = False
            for ln in range(s.start_line, s.end_line + 1):
                if not (1 <= ln <= file_len):
                    continue
                in_bounds = True
                text_exp, col_map = expand(ln)
                x0 = col_map[s.start_col - 1] if ln == s.start_line else 1
                x1 = col_map[min(s.end_col, len(col_map)) - 1] if ln == s.end_line else len(text_exp) + 1
                by_line.setdefault(ln, []).append((lab, x0, x1, ln == s.start_line))
            if not in_bounds:
                dangling.append(lab)

        last = None
        for ln in show:
            if last is not None and ln != last + 1:
                gap = "…" if self.opt.use_unicode else "..."
                lines.append(f"{self.theme.c('gutter', str(gap).rjust(width))} |" if gutter_on else gap)
            last = ln

            exp, _ = expand(ln)
            if gutter_on:
                g = self.theme.c('gutter', str(ln).rjust(width))
                lines.append(f" {g} | {exp}")
            else:
                lines.append(exp)

            if ln in by_line:
                for lab, x0, x1, first_line in sorted(by_line[ln], key=lambda t: 0 if t[0].kind == 'primary' else 1):
                    underline = '^' if lab.kind == 'primary' else '~'
                    if x1 <= x0:
                        x1 = x0 + 1
                    guide = _underline(exp, x0, x1, underline)
                    if gutter_on:
                        pad = self.theme.c('gutter', ' '.rjust(width + 1) + ' | ')
                        row = pad + self.theme.c(lab.kind, guide)
                    else:
                        row = self.theme.c(lab.kind, guide)
                    if lab.message and first_line:
                        row += ' ' + lab.message
                    lines.append(row)

        for lab in dangling:
            s = lab.span.normalized()
            kind = self.theme.c(lab.kind, lab.kind)
            msg = f" {lab.message}" if lab.message else ""
            lines.append(f"{g_head} (span out of range) {path}:{s.start_line}:{s.start_col}-{s.end_line}:{s.end_col} {kind}{msg}")

        return "\n".join(lines)

def _group_by_file(labels: Sequence[Label]) -> Dict[str, List[Label]]:
    buckets: Dict[str, List[Label]] = {}
    for l in labels:
        buckets.setdefault(l.span.file, []).append(l)

    return buckets

def _expand_tabs_with_map(s: str, tabw: int) -> Tuple[str, List[int]]:
    out, mapping, col = [], [], 1
    for ch in s:
        mapping.append(col)
        if ch == "\t":
            n = tabw - (col - 1) % tabw
            out.append(" " * n)
            col += n
        else:
            out.append(ch)
            col += 1

    mapping.append(col)

    return "".join(out), mapping

def _underline(line: str, x0: int, x1: int, ch: str) -> str:
    n = len(line)
    a, b = max(1, min(x0, n)), max(1, min(x1, n + 1))
    if b <= a:
        b = a + 1

    return " " * (a - 1) + ch * (b - a) + " " * (n - (b - 1))


if __name__ == "__main__":
    provider = MemorySourceProvider(
        {
            "demo.cool": (
                "fun main() {\n"
                "    let x = 1 + ;\n"
                "    print(x)\n"
                "}\n"
            ),
            "multi.cool": (
                "fn compute(a, b, c) {\n"
                "    let total = a +\n"
                "        b + c;\n"
                "    let total = 0;  // shadowing error\n"
                "    return total\n"
                "}\n"
            ),
        }
    )

    d1 = Diagnostic(
        severity="error",
        message="syntax error: expected expression",
        labels=[
            Label(Span("demo.cool", 2, 13, 2, 14), "expected expression", "primary"),
            Label(Span("demo.cool", 2, 15, 2, 16), "after '+'", "secondary"),
        ],
        notes=["help: try putting an identifier or literal"],
    )

    d2 = Diagnostic(
        severity="error",
        message="redefinition of 'total'",
        labels=[
            Label(Span("multi.cool", 4, 9, 4, 14), "'total' redefined here", "primary"),
            Label(Span("multi.cool", 2, 9, 2, 14), "previous definition is here", "secondary"),
        ],
    )

    d3 = Diagnostic(
        severity="warning",
        message="line break after '+' requires an expression",
        labels=[
            Label(Span("multi.cool", 2, 20, 3, 9), "expected expression", "primary"),
            Label(Span("multi.cool", 3, 9, 3, 10), "starts next term", "secondary"),
        ],
    )

    r = TerminalRenderer(RenderOptions(context_lines=1))
    for d in (d1, d2, d3):
        r.render(d, source_provider=provider)
