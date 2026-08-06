"""
rhythm.py

Turns a chronological list[Moment] (from moments.py) into the same list
with start_time/duration filled in, using column position as a time
proxy.

STATED APPROXIMATION, NOT REAL RHYTHM (see PLAN.md)
-----------------------------------------------------
Plain-text tab notation does not encode true rhythm. This derives an
approximate duration from the character-gap to the NEXT moment, scaled
by a fixed seconds-per-column constant. A long dash-run before the next
moment reads as a longer held note. Big spacing gaps at section
boundaries will read as unnaturally long holds — that's a known,
accepted distortion (confirmed against real files: gaps up to ~1000
columns occur at section boundaries), not a bug to chase this cycle.

This runs AFTER moments.py, not before: duration is a property of a
shared instant (a Moment), not of an individual Note. Timing raw
per-string note piles first would let two notes in the same chord end
up with two different durations for what's supposed to be one onset.
"""

from src.moments import Moment


def assign_timing(
    moments: list[Moment],
    seconds_per_column: float = 0.05,
    bpm: float | None = None,
    columns_per_beat: float = 8.0,
) -> list[Moment]:
    """
    Mutates and returns `moments` with start_time/duration filled in:
      start_time = moment.column * seconds_per_column
      duration   = (next moment's column - this moment's column) * seconds_per_column

    Last moment has no "next" moment to measure against, so it falls
    back to the previous moment's duration (hold about as long as the
    note before it did) — or, if there's only one moment total,
    `seconds_per_column * 4` as a last resort default.

    BPM, IF YOU HAVE ONE FOR THE SONG
    ------------------------------------
    `seconds_per_column=0.05` on its own is an arbitrary unitless
    scalar — there's no honest way to know it's right for a given song
    without a reference. Pass `bpm` (the song's real, known tempo) and
    it's used instead: `seconds_per_column = 60 / (bpm *
    columns_per_beat)`.

    `columns_per_beat=8` (an eighth of a beat per tab column) is a
    calibrated DEFAULT, not a fixed truth — checked against Hotel
    California specifically, whose tab has real recording timestamps
    printed in it ("(4:20)", "(4:28)", ...). Measuring real elapsed
    seconds against actual tab-column distance between those markers
    gives ~0.089s/column on average at this song's 75 BPM, which is
    almost exactly 60/(75*9) — i.e. ~9 columns/beat for THIS
    transcriber's dash density, with real per-passage variance (0.03
    to 0.12 s/column across sections; a busier passage is typed with
    relatively fewer dashes per note than a sparse one). 8 is picked as
    a round, close, slightly-conservative default — still an
    approximation (see PLAN.md: true rhythm isn't recoverable from tab
    text), but now anchored to a real number instead of guessed
    outright. Every transcription's dash density will differ somewhat,
    so treat this as a starting point to tune per song, not a
    universal constant.

    If `bpm` is not given, falls back to the raw `seconds_per_column`
    behavior (unchanged, for backward compatibility).
    """
    if bpm is not None:
        seconds_per_column = 60.0 / (bpm * columns_per_beat)

    if not moments:
        return moments

    for i, moment in enumerate(moments):
        moment.start_time = moment.column * seconds_per_column

        if i + 1 < len(moments):
            moment.duration = (moments[i + 1].column - moment.column) * seconds_per_column
        elif i > 0:
            moment.duration = moments[i - 1].duration
        else:
            moment.duration = seconds_per_column * 4

    return moments


if __name__ == "__main__":
    from src.parser import Note

    sample_notes = [
        Note(string_index=0, string_name="e", fret=5, column=2),
    ]
    sample_moments = [
        Moment(notes=sample_notes, column=2),
        Moment(notes=sample_notes, column=10),
        Moment(notes=sample_notes, column=14),
    ]

    for m in assign_timing(sample_moments):
        print(f"Moment(col={m.column}): start={m.start_time:.2f}s duration={m.duration:.2f}s")
