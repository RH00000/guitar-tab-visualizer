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
- **Barre chords** (one finger covering multiple strings at once) — the
  optimizer's whole model is "one finger, one note." Barre chords break
  that assumption structurally, not as an edge case. Excluded on
  purpose, matches the actual target use case (soloing, not chord
  voicings — confirmed as the real goal, not an assumption)
- Dense 3+ note chord voicings — the optimizer only needs to handle
  single notes and simple 2-note dyads, which is what solo tab actually
  contains
- Web frontend — stays a local Python script for now

### Handled, not excluded (surfaced this session, worth naming explicitly)
- **Open strings (fret 0):** need no finger at all. `HandPosition.finger`
  can be `None` specifically for this case.
- **Same-string collisions during moment-grouping:** two notes that would
  otherwise bundle into one moment get rejected from merging if they
  share a string — physically impossible, so it's almost certainly noisy
  tab formatting, and gets flagged with a warning instead of silently
  merged or silently crashing.

## Architecture

```
raw tab text (.txt)
      │
      ▼
  parser.py        →  list[Note]                           (DONE, verified against real file)
      │
      ▼
  moments.py        →  list[Moment]  (notes grouped by shared onset)   (NEXT — was a missing piece, not a rename)
      │
      ▼
  rhythm.py         →  list[Moment] with start_time/duration added     (after moments — order matters, see below)
      │
      ▼
  optimizer.py      →  one chosen joint finger-assignment per Moment   (after rhythm)
      │
      ▼
  visualizer.py     →  animated fretboard, matplotlib                  (last)
      │
      ▼
  main.py ties all five together
```

**Corrected order note:** rhythm must run AFTER moments, not before.
Duration belongs to a moment (one shared instant), not to an individual
note — running rhythm on raw per-string note piles first would let two
notes in the same chord end up with two different durations for what's
supposed to be a single shared onset. This was caught and fixed before
any code was written for either module — see METHODS_AND_INTERFACES.md
for the full reasoning.

## Milestones

| # | Deliverable | Depends on | Est. time |
|---|---|---|---|
| 1 | Parser correctness (multi-digit frets, real labels, sanity checks) | — | **DONE** |
| 2 | `moments.py`: group notes into shared-onset moments, with same-string collision guard | Parser | 1 day |
| 3 | `rhythm.py`: moment column-gap → start_time/duration | Moments | 0.5 day (simpler than originally scoped, see METHODS_AND_INTERFACES.md) |
| 4 | Optimizer: candidate joint assignments per moment (handles chords/dyads, not just single notes), cost function, DP shortest path | Rhythm | 3-4 days |
| 5 | Visualizer: static fretboard render, then animate joint assignments over time | Optimizer | 1-2 days |
| 6 | Integration (`main.py`), test against 2-3 different real tabs, README write-up including all stated limitations | Everything | 1 day |

**Total: ~8-9 working days.** Spread over 2 weeks if working alongside
coursework, not 2-3 days.

## Round of play — what actually happens, in order, on a real run

Walking `main.py` running against the Hotel California solo, step by
step, no code — just the logic:

1. **Read the file.** The whole `.txt` becomes one big block of text in
   memory. Nothing clever yet.

2. **Parse it.** The text gets broken into 6-line chunks (blocks), each
   chunk's 6 lines get read character by character, and every real note
   found becomes one small record: which string, which fret, which
   character-column it started at, and any hammer/pull/slide/bend/
   vibrato attached to it. End result: one long list of ~289 of these
   records, in roughly the order they appear in the song.

3. **Group into moments.** Walk that list once, ordered by column. Notes
   that land close enough together in column position get bundled into
   one "moment" — most moments end up being just 1 note (a single-note
   solo line), but a few are 2 notes (an actual chord, like the G-fret-7
   + D-fret-9 dyad found in the real file). If two notes that would
   otherwise bundle together happen to be on the SAME string, that's
   rejected — a string can't play two frets at once, so that's almost
   certainly noisy formatting, not a real chord, and it gets flagged
   rather than silently merged.

4. **Assign timing.** Walk the moment list once. Each moment's length in
   time is set by how far away the NEXT moment is — a big gap in the
   original tab (lots of dashes) becomes a longer held note; a tight
   gap becomes a short one. This is a deliberate approximation, not real
   rhythm — stated on purpose, not hidden.

5. **Solve fingering.** This is the scoreboard step. Starting from
   moment 1, for every moment, work out every physically reasonable way
   a hand could be positioned to play it (which finger, or fingers if
   it's a chord). Instead of committing to a choice immediately, keep a
   running tally of "cheapest way to have reached THIS hand-shape, at
   THIS moment, having come from any possible hand-shape at the
   PREVIOUS moment." Carry that tally all the way to the last moment.
   Only then look back through the tally to recover the actual cheapest
   sequence of finger choices for the whole song — which is exactly how
   a note early in the solo can end up "choosing" a finger based on
   what happens several notes later, without ever explicitly looking
   ahead itself.

6. **Draw it.** Draw the 6-string, N-fret grid once. Then, moment by
   moment, using the timing from step 4 and the finger choices from
   step 5, place a colored dot (or dots, for a chord) at the right
   spot on that grid, and play the moments back at their real timed
   pace — dots jump between frets for normal notes, slide partway for
   bends, and wiggle in place for vibrato.

That's the whole run, start to finish, with every step now checked
against the actual bugs this project has already surfaced (column
misalignment, chord ordering, open strings, same-string collisions) —
not a clean-room description of how it would work if the data were
perfect.

## Known limitations to state explicitly in README (not hide)
- Column position is an ordinal time proxy, not real timing; durations
  are a derived heuristic from spacing, not extracted rhythm
- Multi-digit frets can misalign "simultaneous" notes across strings by
  a column or two (mitigated in milestone 3, not eliminated)
- Grace notes, ties, and some rare notation (e.g. `r` release-bend) are
  not fully modeled
