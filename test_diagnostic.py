import io
import pytest

from diagnostic import (
    Span,
    Label,
    Diagnostic,
    MemorySourceProvider,
    TerminalRenderer,
    RenderOptions,
    TerminalTheme,
    _underline,
    _group_by_file,
)

def test_underline_primary():
    line = "abcdef"
    res = _underline(line, 3, 5, "^")
    assert res == "  ^^  "

def test_group_by_file():
    lbl1 = Label(Span("foo.cool", 1, 1, 1, 2))
    lbl2 = Label(Span("bar.cool", 1, 1, 1, 2))
    groups = _group_by_file([lbl1, lbl2])
    assert set(groups.keys()) == {"foo.cool", "bar.cool"}
    assert groups["foo.cool"][0] is lbl1

def test_span_normalized_swaps_if_needed():
    s = Span("a", 5, 10, 3, 2)
    n = s.normalized()
    assert (n.start_line, n.start_col, n.end_line, n.end_col) == (3, 2, 5, 10)

@pytest.fixture
def sample_renderer():
    opts = RenderOptions(context_lines=0)
    theme = TerminalTheme(use_color=False)
    return TerminalRenderer(opts, theme)

@pytest.fixture
def sample_provider():
    return MemorySourceProvider(
        {
            "demo.cool": "fun main() {\n    let x = 1 + ;\n    print(x)\n}\n",
            "multi.cool": (
                "fn compute(a, b, c) {\n"
                "    let total = a +\n"
                "        b + c;\n"
                "    let total = 0;\n"
                "    return total\n"
                "}\n"
            ),
        }
    )

def test_basic_error_message(sample_renderer, sample_provider):
    diag = Diagnostic(
        severity="error",
        message="syntax error: expected expression",
        labels=[
            Label(Span("demo.cool", 2, 13, 2, 14), "expected expression", "primary"),
            Label(Span("demo.cool", 2, 15, 2, 16), "after '+'", "secondary"),
        ],
        notes=["help: try putting an identifier or literal"],
    )

    out = sample_renderer.format(diag, sample_provider)

    assert "error: syntax error: expected expression" in out
    assert "demo.cool:2:13" in out
    assert "^     expected expression" in out
    assert "= note help: try putting an identifier or literal" in out

def test_redefinition_warning(sample_renderer, sample_provider):
    diag = Diagnostic(
        severity="error",
        message="redefinition of 'total'",
        labels=[
            Label(Span("multi.cool", 4, 9, 4, 14), "'total' redefined here", "primary"),
            Label(Span("multi.cool", 2, 9, 2, 14), "previous definition is here", "secondary"),
        ],
    )
    out = sample_renderer.format(diag, sample_provider)
    assert "'total' redefined here" in out
    assert "previous definition is here" in out
    assert out.count("total") >= 2

def test_render_writes_to_stream(sample_provider):
    diag = Diagnostic(severity="warning", message="dummy")
    renderer = TerminalRenderer(RenderOptions(), TerminalTheme(use_color=False))
    buf = io.StringIO()
    renderer.render(diag, out=buf, source_provider=sample_provider)
    assert "warning: dummy" in buf.getvalue()
