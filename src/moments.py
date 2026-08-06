"""
moments.py

Groups the flat, unordered-by-string list[Note] produced by parser.py
into Moments: notes that sound at the same instant (a single note, most
of the time, or a chord/dyad).

WHY THIS IS ITS OWN STEP
-------------------------
A chord's notes can't get a fingering decided independently, note by
note — you can't use the same finger twice, and the whole chord has to
be reachable by one hand at once. The optimizer needs to make ONE joint
decision per shared instant, not one decision per note. That "shared
instant" is a Moment.

Duration also belongs to a Moment, not an individual Note — see
rhythm.py, which runs AFTER this module, not before.
"""

from dataclasses import dataclass

from src.parser import Note


@dataclass
class Moment:
    notes: list[Note]
    column: int                        # earliest note's column in this moment
    start_time: float | None = None    # filled in later, by rhythm.py
    duration: float | None = None      # filled in later, by rhythm.py


def group_into_moments(notes: list[Note], tolerance: int = 1) -> list[Moment]:
    """
    Sort notes by column, then walk once, greedily bundling a note into
    the CURRENT moment if it's within `tolerance` columns of that
    moment's FIRST note.

    Why anchor to the moment's first note, not the previously-added
    note: comparing each new note only to the last one added lets the
    window drift note by note. On a real repetitive riff (checked
    against Master of Puppets' palm-muted gallop section) that drift
    chains hundreds of unrelated notes into a single "moment" — a
    correctness bug, not a style choice. Anchoring to the first note
    keeps every moment bounded to a genuine shared instant.

    tolerance default is 1, not 3: the higher tolerance exists to catch
    multi-digit-fret column misalignment between simultaneous notes on
    different strings, but at tolerance=3, checked against all 5 real
    tab files in data/, 62% of resulting groups were false-positive
    same-string collisions (consecutive melody notes on ONE string,
    2-3 columns apart — not chords). tolerance=1 keeps the genuine
    misalignment catch while keeping the collision warning meaningful
    instead of drowning it in noise.

    Built-in guard: a note is rejected from joining the current moment
    if its string_name is already used by a note already in that
    moment — a single string can't sound two frets at once, so that's
    almost certainly noisy tab formatting, not a real chord. The
    rejected note starts a new moment instead, and a warning prints
    (same "tell, don't hide" convention as parser.py's fret-number fix)
    so the raw text at that spot can be checked by hand.
    """
    if not notes:
        return []

    sorted_notes = sorted(notes, key=lambda n: n.column)

    moments: list[Moment] = []
    current_notes: list[Note] = [sorted_notes[0]]
    anchor_column = sorted_notes[0].column
    used_strings: set[str] = {sorted_notes[0].string_name}

    for note in sorted_notes[1:]:
        within_window = note.column - anchor_column <= tolerance
        collides = note.string_name in used_strings

        if within_window and collides:
            print(
                f"WARNING: column {note.column}: note on string "
                f"'{note.string_name}' (fret {note.fret}) would join the "
                f"moment anchored at column {anchor_column}, but that "
                f"moment already has a note on string "
                f"'{note.string_name}' — a string can't sound two frets "
                f"at once, so this is almost certainly noisy tab "
                f"formatting. Starting a new moment instead; verify "
                f"against the original tab."
            )

        if within_window and not collides:
            current_notes.append(note)
            used_strings.add(note.string_name)
        else:
            moments.append(Moment(notes=current_notes, column=anchor_column))
            current_notes = [note]
            anchor_column = note.column
            used_strings = {note.string_name}

    moments.append(Moment(notes=current_notes, column=anchor_column))

    return moments


if __name__ == "__main__":
    # Small hand-checkable sample: one single note, one genuine 2-note
    # chord (same column, different strings), and one same-string
    # collision (two notes on the same string within tolerance, which
    # should NOT merge and should print a warning).
    sample_notes = [
        Note(string_index=0, string_name="e", fret=5, column=2),
        Note(string_index=2, string_name="G", fret=7, column=10),
        Note(string_index=3, string_name="D", fret=9, column=10),
        Note(string_index=0, string_name="e", fret=8, column=20),
        Note(string_index=0, string_name="e", fret=10, column=21),
    ]

    for m in group_into_moments(sample_notes):
        print(f"Moment(col={m.column}): {[(n.string_name, n.fret) for n in m.notes]}")
