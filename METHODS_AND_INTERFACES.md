# Methods and Interfaces

**Interface** = the contract: name, what goes in, what comes out, what it
does. Applies to anything.
**Method** = specifically a function that lives on a class (needs an
object to call it on, e.g. `optimizer.solve(...)`).
Parser, moments, and rhythm below are standalone functions — interfaces,
not methods. Optimizer and visualizer are classes — theirs are methods.

**Corrected pipeline order (this fixes a real bug from the previous
version):**

```
parser.py  →  moments.py  →  rhythm.py  →  optimizer.py  →  visualizer.py
```

Why moments has to come BEFORE rhythm, not after: duration is a property
of a MOMENT (one shared onset in time), not of an individual note. If
rhythm ran on raw per-string note piles first, two notes in the same
chord could end up with two different durations for what's supposed to
be one shared instant. Grouping first, then timing the groups, avoids
that entirely.

---

## 1. `parser.py` — DONE, unchanged

Same as before: `Note` dataclass (`string_index`, `string_name`, `fret`,
`column`, `arrival`, `modifier`, `bend_target`), plus `is_tab_line`,
`group_into_blocks`, `parse_line`, `parse_block`, `parse_tab`.
Output: `list[Note]` — flat, unordered-by-string, one entry per real note
played, in the order the parser encountered them.

---

## 2. `moments.py` — NEW MODULE (this didn't exist in the last version — it was a real gap, not a rename)

### Class: `Moment` (dataclass — plain container, same spirit as `Note`)
| Field | Type | Meaning |
|---|---|---|
| `notes` | `list[Note]` | 1 note (the normal case) or 2+ (a chord/dyad) |
| `column` | `int` | representative column for this moment (the earliest note's column) |
| `start_time` | `float \| None` | filled in later, by rhythm.py |
| `duration` | `float \| None` | filled in later, by rhythm.py |

### Function: `group_into_moments(notes: list[Note], tolerance: int = 3) -> list[Moment]`
- Input: the flat note list from `parse_tab`, and a tolerance (how many
  columns apart two notes can be and still count as "the same instant" —
  exists specifically because of the multi-digit-fret misalignment issue
  confirmed on your real file, not a made-up number)
- Output: notes collapsed into `Moment`s, in chronological order
- Does: sorts all notes by column, walks through once, and greedily adds
  a note to the current moment if it's within `tolerance` columns of it;
  otherwise closes the moment and starts a new one
- **Built-in guard, not a nice-to-have:** if two notes about to join the
  same moment share the same `string_name`, they're rejected from
  merging — a single string physically cannot sound two frets at once,
  so this is almost certainly noisy tab formatting, not a real chord.
  The second note starts a new moment instead, and a warning prints so
  you can go check the raw text at that spot (same "tell, don't hide"
  approach as the fret-number fix in the parser).

---

## 3. `rhythm.py` — REVISED (now operates on Moments, not raw per-string note lists)

### Function: `assign_timing(moments: list[Moment], seconds_per_column: float = 0.05) -> list[Moment]`
- Input: the moment list from `group_into_moments`, and a tempo constant
- Output: same moments, `start_time` and `duration` filled in
- Does: `start_time = moment.column * seconds_per_column`.
  `duration = (next moment's column − this moment's column) * seconds_per_column`.
  Last moment gets a fallback default duration (no "next" to measure against).
- **This got simpler after the fix**, not more complicated — the old
  version needed a private per-string grouping helper just to figure out
  "what's the next note on this string." Once moments already represent
  one shared instant in time across all strings, that helper isn't
  needed at all. Correcting the bug removed code, it didn't add any.

---

## 4. `optimizer.py` — REVISED (the real fix: chords need JOINT decisions, not one independent decision per note)

This is the part your "how do you decide the finger without looking at
neighbors" question and the G-fret-7/D-fret-9 chord example both point
at. A chord's notes can't be assigned fingers independently of each
other — you can't use the same finger twice, and the whole chord has to
be reachable by one hand at once. So the graph's nodes have to be
**whole-moment assignments**, not one Note at a time.

### Class: `HandPosition` (dataclass, one finger's assignment for one note)
| Field | Type | Meaning |
|---|---|---|
| `note` | `Note` | which note this covers |
| `finger` | `int \| None` | 1=index, 2=middle, 3=ring, 4=pinky; `None` for open strings (fret 0 needs no finger at all — a real case, not an edge case to special-case away) |
| `anchor_fret` | `int \| None` | implied by `fret − (finger − 1)`; `None` when finger is `None` |

### Class: `FingeringOptimizer`

**Class variables (config):** `stretch_weight`, `shift_weight`,
`max_stretch_frets` — unchanged from before.

**`__init__(self, stretch_weight=..., shift_weight=..., max_stretch_frets=4)`**

**`candidate_assignments(self, moment: Moment) -> list[list[HandPosition]]`**
- Input: one moment (1 or more notes)
- Output: every physically valid way to assign fingers to ALL notes in
  this moment at once. Each item in the returned list is one full
  assignment — for a single-note moment, that's just 1-4 options same as
  before. For a 2-note moment, this generates finger-choice COMBINATIONS
  across both notes together, then throws out any combination that
  reuses a finger, or assigns an open-string note (fret 0) a finger at
  all.
- **Distance used for feasibility must be physical, not raw fret-count.**
  Frets get narrower toward the body (~94% the width of the previous
  fret each step, standard equal-tempered spacing) — a 2-fret stretch at
  fret 3 is a real reach; a 2-fret stretch at fret 15 is nearly nothing.
  Using raw `abs(fret_a - fret_b)` would treat both as equally hard,
  which is wrong and would bias the optimizer away from upper-fret
  passages that are actually easy. Needs a fret→physical-distance
  conversion (lookup table or formula using that ~0.943 ratio) before
  any stretch/feasibility math runs on fret numbers — the exact
  table/formula is tomorrow's work, but every function touching fret
  distance needs to be built assuming physical units, not fret-count
  units, from the start.
- Does NOT need to handle: barre chords, 3+ note dense voicings — those
  are explicitly out of scope (see PLAN.md exclusions). This only needs
  to cover single notes and simple 2-note dyads, which matches the
  actual solo material this project targets.

**`transition_cost(self, from_assignment: list[HandPosition], to_assignment: list[HandPosition], rest_time: float) -> float`**
- Input: two consecutive moments' full joint assignments, AND the amount
  of rest time between them (previous moment's end → next moment's
  start)
- Output: one cost number
- Does: the real math — **not designed yet, next session's actual work**.
  But two shape decisions are locked in now, not deferred:
  1. Cost must be discounted when `rest_time` is large, since a big hand
     reposition during a long pause is nearly free for a real player,
     while the same jump between two rapid-fire notes is genuinely hard.
  2. Anchor/fret distance must be measured in PHYSICAL units (accounting
     for frets narrowing toward the body, ~0.943 ratio per fret), not
     raw fret-count — same reasoning as the `candidate_assignments`
     feasibility check above. A jump from fret 3→5 and a jump from fret
     15→17 are the same fret-count distance but very different real
     stretches; treating them as equal would bias the optimizer away
     from upper-fret passages that are actually easy to play.
  Without these two inputs, the optimizer would systematically misprice
  both fast/slow transitions and low/high-fret transitions — structural
  bugs, not tuning details, which is why they're fixed in the signature
  today rather than left for the cost-function session.

**`solve(self, moments: list[Moment]) -> list[list[HandPosition]]`**
- Input: the full timed moment list
- Output: one chosen joint assignment per moment — the globally cheapest
  path through the whole song
- Does: DP where each LAYER is a moment (not a note), each NODE is one
  full joint assignment for that moment (from `candidate_assignments`),
  edges priced by `transition_cost` — which needs the rest time between
  each pair of consecutive moments passed in, computed from their
  `start_time`/`duration`. Same forward-compute, backward-trace shape as
  before — nothing about the DP mechanism itself changed, only what a
  "node" represents and what `transition_cost` needs to know.

---

## 5. `visualizer.py` — REVISED (signature only, to match Moments)

### Class: `FretboardVisualizer`
Same instance variables as before (`num_frets`, `fig`, `ax`).

**`__init__(self, num_frets: int = 15)`**
**`draw_static_fretboard(self) -> None`**

**`animate(self, moments: list[Moment], assignments: list[list[HandPosition]]) -> matplotlib.animation.FuncAnimation`**
- Input: timed moments (for `start_time`/`duration`) and the optimizer's
  chosen joint assignment per moment (parallel lists — same length,
  same order)
- Output: a `FuncAnimation`
- Does: per frame, draws one dot per `HandPosition` in the current
  moment's assignment (so chords draw multiple dots at once, naturally,
  with no special-casing needed here — the work already happened
  upstream)

**`save(self, animation, path: str) -> None`** — unchanged.