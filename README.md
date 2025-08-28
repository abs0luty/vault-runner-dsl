## Vault runner DSL for AI1030

### 1. Lexical structure

| Category          | Pattern / rule     | Notes                                                                                                           |
| ----------------- | ------------------ | --------------------------------------------------------------------------------------------------------------- |
| **Whitespace**    | `␠ ␉ ␍ ␊ …`        | Ignored except to separate tokens.                                                                              |
| **Line-comment**  | `// … ⏎`           | Skipped by the lexer.                                                                                           |
| **Block-comment** | `/* … */`          | Nesting **not** supported. May span lines.                                                                      |
| **Symbols**       | `{  }  ;`          | Each is a distinct token.                                                                                       |
| **Identifier**    | `[a-z_][a-z0-9_]*` | All keywords are written in lower-case snake\_case and are reserved. There are no user-defined identifiers yet. |
| **End-of-file**   | synthetic          | Produced once by the lexer.                                                                                     |

Reserved words (cannot be used as identifiers)

```
if then else while not break
move turn_left turn_right random_turn pick_key open_door
front_clear on_key at_door at_exit
```

---

### 2. Grammar (EBNF)

```
program        ::= { statement } EOF ;

statement      ::= action ";"                -- primitive step
                 | conditional
                 | loop
                 | "break" ";"               -- exits nearest loop
                 ;

action         ::= "move"
                 | "turn_left"
                 | "turn_right"
                 | "random_turn"
                 | "pick_key"
                 | "open_door"
                 ;

conditional    ::= "if" [ "not" ] condition "then" block
                    [ "else" block ] ;

loop           ::= "while" [ "not" ] condition "then" block ;

block          ::= "{" { statement } "}" ;

condition      ::= "front_clear"
                 | "on_key"
                 | "at_door"
                 | "at_exit" ;
```

*Terminals* are quoted.
*Non-terminals* are lower-case identifiers.
`{ … }` means **zero or more**, `[ … ]` means **optional**, `|` is choice.

---

### 3. Execution model

| Aspect                  | Definition                                                                                                                                                                                                                                                                                         |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **World**               | Rectangular grid surrounded by *walls* (`#`). Default implementation uses a **square** with `size×size` walkable tiles (size ≥ 1). Exactly one *key* (`K`), one *door* (`D`), and one *exit* (`E`) are placed on distinct floor tiles.                                                             |
| **Robot**               | Occupies exactly one tile, faces one of the four compass directions **N E S W**. Internal flag `has_key`.                                                                                                                                                                                          |
| **Step counter**        | Increments after every successful `action`.                                                                                                                                                                                                                                                        |
| **Program termination** | • Normal: reaches `exit` **or** successfully executes `open_door`. <br>• Early: `break` leaves the innermost loop; if control reaches end of `program`, execution stops.<br>• Runtime error: invalid action (e.g.\ move into wall, open door without key) aborts execution and emits a diagnostic. |

---

### 4. Semantic rules

* **move** — If `front_clear` is *true*, advance one tile forward; otherwise runtime error.

* **turn\_left / turn\_right** — Rotate 90° CCW / CW.

* **random\_turn** — Non-deterministically performs *turn\_left* or *turn\_right*.

* **pick\_key** — Succeeds only if the robot stands on `K`; sets `has_key := true` and converts that tile to `floor`.

* **open\_door** — Succeeds only if robot stands on `D` **and** `has_key = true`; converts door tile to `floor`, sets internal flag `door_opened := true`.

* **Conditions**

  | Name          | Truth value                                |
  | ------------- | ------------------------------------------ |
  | `front_clear` | The tile directly ahead is **not** a wall. |
  | `on_key`      | Current tile contains `K`.                 |
  | `at_door`     | Current tile contains `D`.                 |
  | `at_exit`     | Current tile contains `E`.                 |

* The keyword **not** in front of a condition negates its truth value.

* `if`/`while` use *short-circuit* evaluation; conditions have no side effects.

---

### 5. Error handling & diagnostics

Errors are detected at three levels:

| Stage         | Example error                               | Handling                                                             |
| ------------- | ------------------------------------------- | -------------------------------------------------------------------- |
| **Lexical**   | Illegal character, unterminated `/* …`      | Abort, diagnostic points at offending byte.                          |
| **Syntactic** | Missing `;`, `}`                            | Abort, diagnostic shows primary span and optional secondary hints.   |
| **Runtime**   | Moving into a wall, `open_door` without key | Abort, diagnostic emitted with current source location if available. |

All diagnostics are rendered with `diagnostic.py`, giving Rust-style code frames, coloured output when the terminal supports ANSI, and optional notes. Example (`examples/syntax_error.vl`):

```
error: expected 'then'
  --> examples/syntax_error.vl:4:16
   |
 3 |     if on_key  then { pick_key; }
 4 |     if at_door { open_door; }
   |                ^             
 5 |     if at_exit then { break; }
```

---

### 6. Comment syntax

* **Line comment** `// text … ⏎` – ignored to end of line
* **Block comment** `/* text … */` – may span lines, cannot be nested

Comments are treated as whitespace; they may appear anywhere a token boundary is legal.

---

### 7. Minimal example

```cool
/* find key, open the door, or leave via exit */
while true then {
    if on_key  then { pick_key; }
    if at_door then { open_door; }
    if at_exit then { break; }
    if front_clear then { move; } else { random_turn; }
}
```

*Execution ends* with a success message once either the door is opened or the exit tile is reached.
