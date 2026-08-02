# Guitar Tab Visualizer — Project Plan

## Goal
Portfolio project demonstrating applied math + software engineering for
AI/ML internship applications. Takes a plain-text guitar tab, parses it,
computes an efficient hand-fingering path (shortest-path / DP over hand
positions), and renders an animated visualization of a hand playing it.

## Scope decisions (locked in — revisit only with a real reason)

### Input format: plain text tab (Ultimate Guitar style)
**Decision:** Stay with plain-text tab as input. Do NOT switch to MIDI or
Guitar Pro (.gp) files.
**Why:** Both alternatives trade "finish the math" time for "learn a new
file format" time. Plain text is free, has no access friction, and the
parser is already 90% correct against a real messy tab (Hotel California
solo, verified against actual output, not assumed).

### Rhythm: derived heuristic, not real transcription
**Decision:** Tab notation does not encode true rhythm. We derive an
*approximate* duration per note from the character-gap to the next note
on the same string, scaled by a fixed "seconds per column" constant.
**Why honest, not fake:** This is NOT claimed to be accurate rhythm. It's
a stated, deliberate approximation — documented as such in the README.
A long dash-run before the next note = held longer. Big spacing gaps
across a whole line (e.g. at a section boundary) will read as unnaturally
long holds — this is a known, accepted distortion, not a bug to chase.
**Concretely:** every `Note` gets a derived `start_time` and `duration` in
seconds, computed in a new small module, from existing `column` data.
No new input format, no new parsing.

### Explicitly OUT of scope (do not build, do not scope-creep into)
- True rhythm/tempo extraction from tab text (not information tab
  notation contains)
- 7-string / drop-tuning auto-detection beyond what falls out naturally
- Grace notes in parentheses `(12)` — currently silently absorbed as a
  normal note; documented limitation, not fixed this cycle
- Web frontend — stays a local Python script for now

## Architecture

```
raw tab text (.txt)
      │
      ▼
  parser.py        →  list[Note]                    (DONE, verified against real file)
      │
      ▼
  rhythm.py         →  list[Note] with start_time/duration added   (NEXT)
      │
      ▼
  optimizer.py      →  chosen (finger, hand-position) per Note      (after rhythm)
      │
      ▼
  visualizer.py     →  animated fretboard, matplotlib               (last)
      │
      ▼
  main.py ties all four together
```

## Milestones

| # | Deliverable | Depends on | Est. time |
|---|---|---|---|
| 1 | Parser correctness (multi-digit frets, real labels, sanity checks) | — | **DONE** |
| 2 | `rhythm.py`: column-gap → start_time/duration | Parser | 1 day |
| 3 | Chord/simultaneity grouping (cluster notes into "same moment" buckets, since exact column equality is unreliable — confirmed on real data) | Parser | 1 day |
| 4 | Optimizer: candidate hand positions per note, cost function, DP shortest path | Rhythm + grouping | 3-4 days |
| 5 | Visualizer: static fretboard render, then animate hand positions over time | Optimizer | 1-2 days |
| 6 | Integration (`main.py`), test against 2-3 different real tabs (not just Hotel California), README write-up including all stated limitations | Everything | 1 day |

**Total: ~8-9 working days.** Spread over 2 weeks if working alongside
coursework, not 2-3 days.

## Known limitations to state explicitly in README (not hide)
- Column position is an ordinal time proxy, not real timing; durations
  are a derived heuristic from spacing, not extracted rhythm
- Multi-digit frets can misalign "simultaneous" notes across strings by
  a column or two (mitigated in milestone 3, not eliminated)
- Grace notes, ties, and some rare notation (e.g. `r` release-bend) are
  not fully modeled
