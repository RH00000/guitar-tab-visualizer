"""
moments.py

Groups the flat, unordered-by-string list[Note] produced by parser.py
into Moments: notes that sound at the same instant (a single note, most
of the time, or a chord/dyad).

Why is this its own step?
-------------------------
A chord's notes can't get a fingering decided independently, note by
note — you can't use the same finger twice, and the whole chord has to
be reachable by one hand at once. The optimizer needs to make ONE joint
decision per shared instant, not one decision per note. That "shared
instant" is a Moment. basically 6 strings together. 

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
    Groups notes into Moments (a shared instant) by walking notes
    sorted by column, bundling into the current moment if within
    `tolerance` columns of that moment's FIRST note -- anchoring to
    the first note, not the last one added, avoids column drift
    chaining unrelated notes together on a repetitive riff. Rejects a
    note whose string is already used in the current moment (a string
    can't sound two frets at once) -- starts a new moment instead and
    warns.
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
        # start a new moment if either the note is too far away in time (column)
        else:
            moments.append(Moment(notes=current_notes, column=anchor_column))
            current_notes = [note]
            anchor_column = note.column
            used_strings = {note.string_name}

    moments.append(Moment(notes=current_notes, column=anchor_column))

    return moments


def previous_note_by_string(moments: list["Moment"]) -> dict[int, Note]:
    """
    Maps id(note) -> the note that came right before it on the SAME
    string, walking moments in chronological order. A note with
    nothing before it on its string is absent from the result.
    Centralized here, not duplicated in visualizer.py/audio.py, since
    both need "previous note on this string" for arrival techniques
    (hammer-on/pull-off/slide).
    """
    last_by_string: dict[int, Note] = {}
    result: dict[int, Note] = {}
    for moment in moments:
        for note in moment.notes:
            prev = last_by_string.get(note.string_index)
            if prev is not None:
                result[id(note)] = prev
            last_by_string[note.string_index] = note
    return result