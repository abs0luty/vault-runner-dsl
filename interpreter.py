"""
Vault‑Runner DSL Interpreter
"""

from __future__ import annotations

import io
import random
import re
import sys
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple, Union

from diagnostic import (
    Diagnostic,
    Label,
    Span,
    TerminalRenderer,
    RenderOptions,
    MemorySourceProvider,
)

renderer = TerminalRenderer(RenderOptions(context_lines=1))
_SRC_PROVIDER: Optional[MemorySourceProvider] = None

def _span(file: str, tok: "Token") -> Span:
    return Span(file, tok.line, tok.col, tok.line, tok.col + max(1, len(tok.text)))

def _report(diag: Diagnostic):
    """Render diagnostic and abort."""
    renderer.render(diag, source_provider=_SRC_PROVIDER)
    sys.exit(1)

def _diagnose(diag: Diagnostic):
    """Render diagnostic but DO NOT abort (used for non-fatal issues like loop caps)."""
    renderer.render(diag, source_provider=_SRC_PROVIDER)

class TokenKind(Enum):
    ident = auto()
    lbrace = auto()
    rbrace = auto()
    semi = auto()
    eof = auto()

@dataclass
class Token:
    kind: TokenKind
    text: str
    line: int
    col: int

_TOKEN_RE = re.compile(
    r"""
    (?P<comment>//[^\n]*|/\*[\s\S]*?\*/)| # comments first – greedy block/single‑line
    (?P<ws>\s+)|
    (?P<sym>[{};])|
    (?P<ident>[a-z_][a-z0-9_]*)|
    (?P<junk>.)
    """,
    re.VERBOSE,
)

KEYWORDS = {
    "if",
    "do",
    "else",
    "while",
    "not",
    "and",
    "or",
    "move",
    "turn_left",
    "turn_right",
    "random_turn",
    "pick_key",
    "open_door",
    "front_clear",
    "on_key",
    "at_door",
    "at_exit",
    "true",
    "false",
    "break",
}

def lex(src: str, file: str) -> List[Token]:
    out: List[Token] = []
    line = col = 1
    for m in _TOKEN_RE.finditer(src):
        lexeme = m.group(0)
        if m.group("comment"):
            nls = lexeme.count("\n")
            if nls:
                line += nls
                col = 1 + len(lexeme) - (lexeme.rfind("\n") + 1)
            else:
                col += len(lexeme)
            continue
        if m.group("ws"):
            nls = lexeme.count("\n")
            if nls:
                line += nls
                col = 1 + len(lexeme) - (lexeme.rfind("\n") + 1)
            else:
                col += len(lexeme)
            continue
        if m.group("sym"):
            kind = {"{": TokenKind.lbrace, "}": TokenKind.rbrace, ";": TokenKind.semi}[lexeme]
            out.append(Token(kind, lexeme, line, col))
            col += 1
            continue
        if m.group("ident"):
            out.append(Token(TokenKind.ident, lexeme, line, col))
            col += len(lexeme)
            continue
        # junk
        _report(
            Diagnostic(
                severity="error",
                message=f"unexpected character '{lexeme}'",
                labels=[Label(_span(file, Token(TokenKind.ident, lexeme, line, col)), "unexpected here", "primary")],
            )
        )
    out.append(Token(TokenKind.eof, "", line, col))
    return out

@dataclass
class Block:
    stmts: List["Node"]
    lbrace: Optional[Token] = None
    rbrace: Optional[Token] = None

@dataclass
class Action:
    name: str
    tok: Token

class Expr: ...

@dataclass
class Prim(Expr):
    name: str  # front_clear / on_key …

@dataclass
class NotExpr(Expr):
    expr: Expr

@dataclass
class BoolLit(Expr):
    value: bool

@dataclass
class BinExpr(Expr):
    op: str  # 'and' | 'or'
    left: Expr
    right: Expr

@dataclass
class Cond:
    tok_if: Token
    expr: Expr
    do_blk: Block
    else_blk: Optional[Block]
    tok_else: Optional[Token] = None

@dataclass
class Loop:
    tok_while: Token
    cond: Expr
    body: Block

Node = Union[Block, Action, Cond, Loop]

class Parser:
    def __init__(self, toks: Sequence[Token], file: str, max_nesting: int = 3):
        self.toks = list(toks)
        self.i = 0
        self.file = file
        self.max_nesting = max_nesting
        # stack of control-structure tokens for nesting diagnostics
        self.ctrl_stack: List[Token] = []

    def _cur(self) -> Token:
        return self.toks[self.i]

    def _eat(self, kind: TokenKind, err: str) -> Token:
        if self._cur().kind is not kind:
            self._err(err)
        tok = self._cur()
        self.i += 1
        return tok

    def _err(self, msg: str, extra_labels: Optional[List[Label]] = None, notes: Optional[List[str]] = None):
        tok = self._cur()
        labels = [Label(_span(self.file, tok), "here", "primary")]
        if extra_labels:
            # convert first label to primary, move "here" to secondary to keep caller-provided primary
            labels = extra_labels + [Label(_span(self.file, tok), "parsing continued here", "secondary")]
        _report(Diagnostic(severity="error", message=msg, labels=labels, notes=notes or []))

    def parse(self) -> Block:
        stmts = []
        while self._cur().kind is not TokenKind.eof:
            stmts.append(self._stmt())
        return Block(stmts)

    def _stmt(self) -> Node:
        tok = self._cur()
        if tok.kind is TokenKind.ident and tok.text in {
            "move",
            "turn_left",
            "turn_right",
            "random_turn",
            "pick_key",
            "open_door",
        }:
            return self._action()
        if tok.text == "break":
            br = self._eat(TokenKind.ident, "expected 'break'")
            self._eat(TokenKind.semi, "expected ';'")
            return Action("__break__", br)
        if tok.text == "if":
            return self._if()
        if tok.text == "while":
            return self._while()
        self._err("expected statement")

    def _action(self) -> Action:
        tok = self._eat(TokenKind.ident, "expected action name")
        self._eat(TokenKind.semi, "expected ';'")
        return Action(tok.text, tok)

    _prim_names = {"front_clear", "on_key", "at_door", "at_exit", "true", "false"}

    def _cond_expr(self) -> Expr:
        def parse_factor():
            if self._cur().text == "not":
                self._eat(TokenKind.ident, "'not'")
                return NotExpr(parse_factor())
            tok = self._eat(TokenKind.ident, "condition name")
            if tok.text in ("true", "false"):
                return BoolLit(tok.text == "true")
            if tok.text not in self._prim_names:
                self._err(
                    "unknown condition",
                    extra_labels=[Label(_span(self.file, tok), f"'{tok.text}' is not a known condition", "primary")],
                    notes=["valid conditions: front_clear, on_key, at_door, at_exit, true, false"],
                )
            return Prim(tok.text)

        def parse_term():  # 'and'-chain
            node = parse_factor()
            while self._cur().kind is TokenKind.ident and self._cur().text == "and":
                and_tok = self._eat(TokenKind.ident, "'and'")
                node = BinExpr("and", node, parse_factor())
                # (example: could add secondary labels around 'and' if needed)
            return node

        node = parse_term()
        while self._cur().kind is TokenKind.ident and self._cur().text == "or":
            or_tok = self._eat(TokenKind.ident, "'or'")
            node = BinExpr("or", node, parse_term())
        return node

    def _check_nesting_or_error(self, new_tok: Token):
        depth = len(self.ctrl_stack) + 1
        if depth > self.max_nesting:
            # primary: the newest (violating) token
            labels = [Label(_span(self.file, new_tok), f"nesting level {depth} starts here", "primary")]
            # secondary: ancestors from inner to outer (up to 3 for context)
            for anc in reversed(self.ctrl_stack[-3:]):
                labels.append(Label(_span(self.file, anc), "enclosing control starts here", "secondary"))
            _report(
                Diagnostic(
                    severity="error",
                    message=f"maximum nesting depth of {self.max_nesting} exceeded",
                    labels=labels,
                    notes=[
                        f"reduce nested 'if'/'while' blocks to at most {self.max_nesting} levels",
                        "tip: break helpers into separate top-level blocks or use 'break' early exits",
                    ],
                )
            )

    def _if(self) -> Cond:
        tok_if = self._eat(TokenKind.ident, "expected 'if'")
        self._check_nesting_or_error(tok_if)
        self.ctrl_stack.append(tok_if)
        expr = self._cond_expr()
        self._eat(TokenKind.ident, "expected 'do'")
        do_blk = self._block()
        else_blk = None
        tok_else = None
        if self._cur().kind is TokenKind.ident and self._cur().text == "else":
            tok_else = self._eat(TokenKind.ident, "expected 'else'")
            # 'else' does not add to nesting level itself; only its block does
            else_blk = self._block()
        self.ctrl_stack.pop()
        return Cond(tok_if, expr, do_blk, else_blk, tok_else)

    def _while(self) -> Loop:
        tok_while = self._eat(TokenKind.ident, "expected 'while'")
        self._check_nesting_or_error(tok_while)
        self.ctrl_stack.append(tok_while)
        expr = self._cond_expr()
        self._eat(TokenKind.ident, "expected 'do'")
        body = self._block()
        self.ctrl_stack.pop()
        return Loop(tok_while, expr, body)

    def _block(self) -> Block:
        lb = self._eat(TokenKind.lbrace, "expected '{'")
        stmts: List[Node] = []
        while self._cur().kind is not TokenKind.rbrace:
            stmts.append(self._stmt())
        rb = self._eat(TokenKind.rbrace, "expected '}'")
        return Block(stmts, lb, rb)

class Tile(str, Enum):
    floor = "."
    wall = "#"
    key = "K"
    door = "D"
    exit = "E"

DIRS = ["N", "E", "S", "W"]
OFF = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}

@dataclass
class Robot:
    r: int
    c: int
    dir: str = "N"
    has_key: bool = False

@dataclass
class World:
    grid: List[List[Tile]]
    bot: Robot
    door_opened: bool = False
    steps: int = 0

    @classmethod
    def square(cls, size: int = 4) -> "World":
        """Preserve tiny grid functionality."""
        h = w = size
        g = [[Tile.wall] * (w + 2) for _ in range(h + 2)]
        for r in range(1, h + 1):
            for c in range(1, w + 1):
                g[r][c] = Tile.floor

        # key, door, exit in three corners deterministically
        positions = [(1, 1), (1, w), (h, w)]
        key_pos, door_pos, exit_pos = positions
        g[key_pos[0]][key_pos[1]] = Tile.key
        g[door_pos[0]][door_pos[1]] = Tile.door
        g[exit_pos[0]][exit_pos[1]] = Tile.exit
        return cls(g, Robot(1, 1, "E"))

    @classmethod
    def twisting_corridor(
        cls,
        turns: int = 6,
        segment_len: int = 2,
        seed: Optional[int] = None,
    ) -> "World":
        """
        Build a single 1-tile-wide corridor that snakes with 'turns' segments.
        Both ends are blocked by walls.
        Place one key, one door, and one exit along the corridor.
        Initial robot position & direction are unknown (random along corridor).
        """
        rng = random.Random(seed)
        # Compute path coordinates
        # Start with padding walls around
        # Allocate bounding box big enough
        W = H = max(10, 2 * turns * segment_len + 5)
        g = [[Tile.wall for _ in range(W)] for _ in range(H)]

        # Start somewhere near center
        r = H // 2
        c = W // 4
        path: List[Tuple[int, int]] = []
        dir_idx = 1  # start heading East for consistency (E,S,E,S,...) but the initial bot dir will be randomized
        for t in range(turns):
            dr, dc = OFF[DIRS[dir_idx]]
            for _ in range(segment_len):
                r += dr
                c += dc
                if 1 <= r < H - 1 and 1 <= c < W - 1:
                    path.append((r, c))
            # turn 90° alternating to create a twisting pattern
            dir_idx = (dir_idx + 1) % 4 if t % 2 == 0 else (dir_idx - 1) % 4

        # Carve corridor cells
        for (rr, cc) in path:
            g[rr][cc] = Tile.floor

        # Block both ends by ensuring neighbors beyond ends are walls (already walls by default)

        # Choose three distinct interior positions for key/door/exit
        interior = path[1:-1] if len(path) > 3 else path
        if len(interior) < 3:
            # Fallback to a trivial small corridor
            interior = path

        pick = rng.sample(interior, k=min(3, len(interior))) if interior else []
        key_pos = pick[0] if pick else path[len(path) // 3]
        door_pos = pick[1] if len(pick) > 1 else path[len(path) // 2]
        exit_pos = pick[2] if len(pick) > 2 else path[-len(path) // 3 or -2]

        g[key_pos[0]][key_pos[1]] = Tile.key
        g[door_pos[0]][door_pos[1]] = Tile.door
        g[exit_pos[0]][exit_pos[1]] = Tile.exit

        # Randomly place robot along corridor (not on wall), random facing
        bot_r, bot_c = rng.choice(path)
        bot_dir = rng.choice(DIRS)
        return cls(g, Robot(bot_r, bot_c, bot_dir))

    def _ahead(self):
        dr, dc = OFF[self.bot.dir]
        return self.bot.r + dr, self.bot.c + dc

    def on_key(self):
        return self.grid[self.bot.r][self.bot.c] == Tile.key

    def at_exit(self):
        return self.grid[self.bot.r][self.bot.c] == Tile.exit

    def move(self):
        if not self.front_clear():
            _report(
                Diagnostic(
                    severity="error",
                    message="cannot move into wall",
                    labels=[
                        Label(Span("<runtime>", 0, 0, 0, 1), "blocked ahead", "primary"),
                    ],
                )
            )
        self.bot.r, self.bot.c = self._ahead()

    def turn_left(self):
        self.bot.dir = DIRS[(DIRS.index(self.bot.dir) - 1) % 4]

    def turn_right(self):
        self.bot.dir = DIRS[(DIRS.index(self.bot.dir) + 1) % 4]

    def random_turn(self):
        random.choice([self.turn_left, self.turn_right])()

    def pick_key(self):
        if not self.on_key():
            _report(
                Diagnostic(
                    severity="error",
                    message="no key here",
                    labels=[Label(Span("<runtime>", 0, 0, 0, 1), "not standing on 'K'", "primary")],
                )
            )
        self.bot.has_key = True
        self.grid[self.bot.r][self.bot.c] = Tile.floor

    def _draw(self) -> str:
        icon = {"N": "↑", "E": "→", "S": "↓", "W": "←"}
        out = []
        for r, row in enumerate(self.grid):
            line = []
            for c, t in enumerate(row):
                line.append(icon[self.bot.dir] if (r, c) == (self.bot.r, self.bot.c) else t.value)
            out.append("".join(line))
        return "\n".join(out)

    def tick(self, action_name: str, from_state: Tuple[int, int, str], to_state: Tuple[int, int, str]):
        self.steps += 1
        fr, fc, fdir = from_state
        tr, tc, tdir = to_state
        print(f"\nstep {self.steps}: {action_name}   [{fdir}@({fr},{fc}) -> {tdir}@({tr},{tc})]")
        print(self._draw())

    def _tile_at(self, r: int, c: int) -> Tile:
        return self.grid[r][c]

    def _tile_ahead(self) -> Tile:
        rr, cc = self._ahead()
        return self.grid[rr][cc]

    def front_clear(self):
        """
        The cell ahead must be traversable.
        Doors are NOT traversable until opened (i.e., changed to floor).
        """
        r, c = self._ahead()
        ahead = self.grid[r][c]
        return ahead not in (Tile.wall, Tile.door)

    def at_door(self):
        """
        True if we are on a door OR facing a door directly ahead.
        This lets programs choose to open from the adjacent tile.
        """
        here = self.grid[self.bot.r][self.bot.c] == Tile.door
        ahead = self._tile_ahead() == Tile.door
        return here or ahead

    def open_door(self):
        """
        Open the door where we stand OR the one directly ahead.
        Requires a key. After opening, the door tile becomes floor.
        """
        # Where is the door?
        here_is_door = self.grid[self.bot.r][self.bot.c] == Tile.door
        rr, cc = self._ahead()
        ahead_is_door = self.grid[rr][cc] == Tile.door

        if not (here_is_door or ahead_is_door):
            _report(
                Diagnostic(
                    severity="error",
                    message="not at door",
                    labels=[
                        Label(Span("<runtime>", 0, 0, 0, 1), "no door here or ahead", "primary"),
                    ],
                )
            )

        if not self.bot.has_key:
            _report(
                Diagnostic(
                    severity="error",
                    message="door locked – need key",
                    labels=[
                        Label(Span("<runtime>", 0, 0, 0, 1), "no key in inventory", "primary"),
                    ],
                    notes=["pick the key first (stand on 'K' and call pick_key)"],
                )
            )

        if ahead_is_door:
            self.grid[rr][cc] = Tile.floor
        else:
            self.grid[self.bot.r][self.bot.c] = Tile.floor

        self.door_opened = True


class Executor:
    def __init__(self, ast: Block, world: World, file: str):
        self.ast = ast
        self.w = world
        self._break = False
        self.file = file
        self._loop_iters: Dict[int, int] = {} 
        self.loop_cap = 50

    def _exec_action(self, act: Action):
        if act.name == "__break__":
            self._break = True
            return
        fn = getattr(self.w, act.name, None)
        if not callable(fn):
            _report(
                Diagnostic(
                    severity="error",
                    message=f"unknown action '{act.name}'",
                    labels=[
                        Label(_span(self.file, act.tok), "invoked here", "primary"),
                    ],
                )
            )
        before = (self.w.bot.r, self.w.bot.c, self.w.bot.dir)
        fn()
        after = (self.w.bot.r, self.w.bot.c, self.w.bot.dir)
        self.w.tick(act.name, before, after)

    def run(self):
        print("initial world")
        print(self.w._draw())
        self._exec_block(self.ast)

    def _eval_expr(self, e: Expr) -> bool:
        if isinstance(e, Prim):
            fn = getattr(self.w, e.name, None)
            if not callable(fn):
                _report(
                    Diagnostic(
                        severity="error",
                        message=f"unknown condition '{e.name}'",
                        labels=[Label(Span("<runtime>", 0, 0, 0, 1), "runtime check", "primary")],
                    )
                )
            return fn()
        if isinstance(e, BoolLit):
            return e.value
        if isinstance(e, NotExpr):
            return not self._eval_expr(e.expr)
        if isinstance(e, BinExpr):
            if e.op == "and":
                return self._eval_expr(e.left) and self._eval_expr(e.right)
            if e.op == "or":
                return self._eval_expr(e.left) or self._eval_expr(e.right)
        _report(Diagnostic(severity="error", message="internal: bad Expr"))
        return False

    def _exec_block(self, blk: Block):
        for node in blk.stmts:
            if self._break:
                return
            if isinstance(node, Action):
                self._exec_action(node)
            elif isinstance(node, Cond):
                val = self._eval_expr(node.expr)
                chosen = node.do_blk if val else node.else_blk
                if chosen:
                    self._exec_block(chosen)
            elif isinstance(node, Loop):
                lid = id(node)
                self._loop_iters[lid] = 0
                while self._eval_expr(node.cond) and not self._break:
                    self._loop_iters[lid] += 1
                    if self._loop_iters[lid] > self.loop_cap:
                        labels = [
                            Label(_span(self.file, node.tok_while), f"loop starts here (>{self.loop_cap} iters)", "primary")
                        ]
                        if node.body.lbrace:
                            labels.append(Label(_span(self.file, node.body.lbrace), "loop body begins here", "secondary"))
                        _diagnose(
                            Diagnostic(
                                severity="error",
                                message=f"loop iteration limit ({self.loop_cap}) reached; breaking out",
                                labels=labels,
                                notes=["guard your loops or add 'break' conditions to avoid infinite running"],
                            )
                        )
                        break
                    self._exec_block(node.body)
            else:
                _report(Diagnostic(severity="error", message="internal: unknown AST node"))

def run_source(
    src: str,
    *,
    file: str = "<input>",
    world_kind: str = "square",  # "square" or "corridor"
    size: int = 4,
    corridor_turns: int = 6,
    corridor_segment_len: int = 2,
    corridor_seed: Optional[int] = None,
) -> World:
    """
    Parse, build world, execute program. Returns the final world for assertions.
    """
    global _SRC_PROVIDER
    _SRC_PROVIDER = MemorySourceProvider({file: src})

    ast = Parser(lex(src, file), file).parse()

    if world_kind == "corridor":
        world = World.twisting_corridor(turns=corridor_turns, segment_len=corridor_segment_len, seed=corridor_seed)
    else:
        world = World.square(size)

    exe = Executor(ast, world, file)
    exe.run()

    print("\n=== program finished ===")
    if world.at_exit():
        print("Robot reached the exit—game complete!")
    elif world.door_opened:
        print("Robot opened the door with the key—game complete!")
    else:
        print("Program ended without reaching exit or opening the door.")
    return world

def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        try:
            with open(path, "r", encoding="utf-8") as fh:
                src = fh.read()
        except OSError as e:
            _report(Diagnostic(severity="error", message=str(e)))
            return

        world_kind = "corridor" if "corridor" in path.lower() else "square"
        run_source(src, file=path, world_kind=world_kind, size=4)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        class TestVaultRunner(unittest.TestCase):
            def run_and_capture(self, src: str, **kwargs) -> str:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    run_source(src, file="<test>", **kwargs)
                return buf.getvalue()

            def test_nesting_limit(self):
                # 4 levels deep -> should error out with multi-label diag
                program = """
                if true do {
                    if true do {
                        if true do {
                            if true do { }
                        }
                    }
                }
                """
                with self.assertRaises(SystemExit):
                    self.run_and_capture(program)

            def test_loop_cap_nonfatal(self):
                # Infinite spin on turns, should hit loop cap (50) then continue to finish
                program = """
                while true do {
                    turn_left;
                }
                """
                out = self.run_and_capture(program)
                self.assertIn("loop iteration limit (50) reached; breaking out", out)
                self.assertIn("=== program finished ===", out)

            def test_corridor_escape(self):
                # Simple wall-follow-ish: move while clear else turn right; grab key if seen; open door; stop at exit/door
                program = """
                while true do {
                    if front_clear do { move; } else { turn_right; }
                    if on_key do { pick_key; }
                    if at_door do { open_door; break; }
                    if at_exit do { break; }
                }
                """
                out = self.run_and_capture(
                    program,
                    world_kind="corridor",
                    corridor_turns=5,
                    corridor_segment_len=2,
                    corridor_seed=42,   # deterministic start/placements
                )
                self.assertRegex(out, r"(Robot reached the exit|Robot opened the door)")
        unittest.main(argv=[""], exit=False)
    else:
        main()
