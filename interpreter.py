"""
Vault‑Runner DSL Interpreter
"""

from __future__ import annotations

import random
import re
import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Sequence, Union

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

def _report(diag: Diagnostic):
    """Render diagnostic and abort."""
    renderer.render(diag, source_provider=_SRC_PROVIDER)
    sys.exit(1)

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
    "then",
    "else",
    "while",
    "not",
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
    "break",
}

def lex(src: str, file: str) -> List[Token]:
    out: List[Token] = []
    line = col = 1
    idx = 0
    for m in _TOKEN_RE.finditer(src):
        lexeme = m.group(0)
        if m.group("comment"):
            # count newlines inside comment for proper line tracking
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
                labels=[Label(Span(file, line, col, line, col + 1), kind="primary")],
            )
        )
    out.append(Token(TokenKind.eof, "", line, col))
    return out

@dataclass
class Block:
    stmts: List["Node"]

@dataclass
class Action:
    name: str
    tok: Token

@dataclass
class Cond:
    cond: str
    negate: bool
    then_blk: Block
    else_blk: Optional[Block]

@dataclass
class Loop:
    cond: str
    negate: bool
    body: Block

Node = Union[Block, Action, Cond, Loop]

class Parser:
    def __init__(self, toks: Sequence[Token], file: str):
        self.toks = list(toks)
        self.i = 0
        self.file = file

    def _cur(self) -> Token:
        return self.toks[self.i]

    def _eat(self, kind: TokenKind, err: str) -> Token:
        if self._cur().kind is not kind:
            self._err(err)
        tok = self._cur()
        self.i += 1
        return tok

    def _err(self, msg: str):
        tok = self._cur()
        _report(
            Diagnostic(
                severity="error",
                message=msg,
                labels=[Label(Span(self.file, tok.line, tok.col, tok.line, tok.col + max(1, len(tok.text))), kind="primary")],
            )
        )

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

    def _if(self) -> Cond:
        self._eat(TokenKind.ident, "expected 'if'")
        negate = False
        if self._cur().text == "not":
            negate = True
            self._eat(TokenKind.ident, "expected 'not'")
        cond_tok = self._eat(TokenKind.ident, "expected condition name")
        self._eat(TokenKind.ident, "expected 'then'")
        then_blk = self._block()
        else_blk = None
        if self._cur().kind is TokenKind.ident and self._cur().text == "else":
            self._eat(TokenKind.ident, "expected 'else'")
            else_blk = self._block()
        return Cond(cond_tok.text, negate, then_blk, else_blk)

    def _while(self) -> Loop:
        self._eat(TokenKind.ident, "expected 'while'")
        negate = False
        if self._cur().text == "not":
            negate = True
            self._eat(TokenKind.ident, "expected 'not'")
        cond_tok = self._eat(TokenKind.ident, "condition name")
        self._eat(TokenKind.ident, "expected 'then'")
        body = self._block()
        return Loop(cond_tok.text, negate, body)

    def _block(self) -> Block:
        self._eat(TokenKind.lbrace, "expected '{'")
        stmts: List[Node] = []
        while self._cur().kind is not TokenKind.rbrace:
            stmts.append(self._stmt())
        self._eat(TokenKind.rbrace, "expected '}'")
        return Block(stmts)

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
        h = w = size
        g = [[Tile.wall] * (w + 2) for _ in range(h + 2)]
        for r in range(1, h + 1):
            for c in range(1, w + 1):
                g[r][c] = Tile.floor

        # we can place key, door, exit in three different corners deterministically
        positions = [(1, 1), (1, w), (h, w)]
        key_pos, door_pos, exit_pos = positions
        g[key_pos[0]][key_pos[1]] = Tile.key
        g[door_pos[0]][door_pos[1]] = Tile.door
        g[exit_pos[0]][exit_pos[1]] = Tile.exit
        return cls(g, Robot(1, 1, "E"))

    def _ahead(self):
        dr, dc = OFF[self.bot.dir]
        return self.bot.r + dr, self.bot.c + dc

    def front_clear(self):
        r, c = self._ahead()
        return self.grid[r][c] != Tile.wall

    def on_key(self):
        return self.grid[self.bot.r][self.bot.c] == Tile.key

    def at_door(self):
        return self.grid[self.bot.r][self.bot.c] == Tile.door

    def at_exit(self):
        return self.grid[self.bot.r][self.bot.c] == Tile.exit

    def move(self):
        if not self.front_clear():
            _report(Diagnostic(severity="error", message="cannot move into wall"))
        self.bot.r, self.bot.c = self._ahead()

    def turn_left(self):
        self.bot.dir = DIRS[(DIRS.index(self.bot.dir) - 1) % 4]

    def turn_right(self):
        self.bot.dir = DIRS[(DIRS.index(self.bot.dir) + 1) % 4]

    def random_turn(self):
        random.choice([self.turn_left, self.turn_right])()

    def pick_key(self):
        if not self.on_key():
            _report(Diagnostic(severity="error", message="no key here"))
        self.bot.has_key = True
        self.grid[self.bot.r][self.bot.c] = Tile.floor

    def open_door(self):
        if not self.at_door():
            _report(Diagnostic(severity="error", message="not at door"))
        if not self.bot.has_key:
            _report(Diagnostic(severity="error", message="door locked – need key"))
        self.grid[self.bot.r][self.bot.c] = Tile.floor
        self.door_opened = True

    def _draw(self) -> str:
        icon = {"N": "↑", "E": "→", "S": "↓", "W": "←"}
        out = []
        for r, row in enumerate(self.grid):
            line = []
            for c, t in enumerate(row):
                line.append(icon[self.bot.dir] if (r, c) == (self.bot.r, self.bot.c) else t.value)
            out.append("".join(line))
        return "\n".join(out)

    def tick(self, action_name: str):
        self.steps += 1
        print(f"\nstep {self.steps}: {action_name}")
        print(self._draw())

class Executor:
    def __init__(self, ast: Block, world: World):
        self.ast = ast
        self.w = world
        self._break = False

    def _eval_cond(self, name: str) -> bool:
        fn = getattr(self.w, name, None)
        if not callable(fn):
            _report(Diagnostic(severity="error", message=f"unknown condition '{name}'"))
        return fn()

    def _exec_action(self, act: Action):
        if act.name == "__break__":
            self._break = True
            return
        fn = getattr(self.w, act.name, None)
        if not callable(fn):
            _report(Diagnostic(severity="error", message=f"unknown action '{act.name}'"))
        fn()
        self.w.tick(act.name)

    def run(self):
        self._exec_block(self.ast)

    def _exec_block(self, blk: Block):
        for node in blk.stmts:
            if self._break:
                return
            if isinstance(node, Action):
                self._exec_action(node)
            elif isinstance(node, Cond):
                val = self._eval_cond(node.cond)
                val = not val if node.negate else val
                chosen = node.then_blk if val else node.else_blk
                if chosen:
                    self._exec_block(chosen)
            elif isinstance(node, Loop):
                while True:
                    val = self._eval_cond(node.cond)
                    val = not val if node.negate else val
                    if not val or self._break:
                        break
                    self._exec_block(node.body)
            else:
                _report(Diagnostic(severity="error", message="internal: unknown AST node"))

def run_source(src: str, *, file="<input>", size: int = 4):
    global _SRC_PROVIDER
    _SRC_PROVIDER = MemorySourceProvider({file: src})

    ast = Parser(lex(src, file), file).parse()
    world = World.square(size)

    print("initial world")
    print(world._draw())
    Executor(ast, world).run()

    print("\n=== program finished ===")
    if world.at_exit():
        print("Robot reached the exit—game complete!")
    elif world.door_opened:
        print("Robot opened the door with the key—game complete!")
    else:
        print("Program ended without reaching exit or opening the door.")

def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        try:
            with open(path, "r", encoding="utf‑8") as fh:
                src = fh.read()
        except OSError as e:
            _report(Diagnostic(severity="error", message=str(e)))
            return
        
        run_source(src, file=path, size=4)

if __name__ == "__main__":
    main()