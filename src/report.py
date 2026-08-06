"""
report.py

Writes the optimizer's solved fingering out as a plain text file --
one line per note, in song order -- so it can be checked by hand
against a real tab/reference instead of only being visible frame by
frame in the animated viewer. Read-only with respect to the rest of
the pipeline: takes the same `moments`/`assignments` every other
module already produces and just formats them, no fingering/cost
logic lives here.

Every run writes one of these (see `next_report_path`), never
overwriting a previous run's report for the same song -- so re-running
main.py after tweaking optimizer weights leaves both the old and new
fingering lists on disk to diff against each other, not just the
latest one silently replacing the last.
"""

import os

from src.moments import Moment
from src.optimizer import HandPosition
from src.parser import Technique


def _describe_note(note) -> str:
    """Fret + technique, e.g. 'fret 7 [BEND->9]' or 'x (muted)'."""
    if note.fret < 0:
        desc = "x (muted)"
    elif note.fret == 0:
        desc = "open string"
    else:
        desc = f"fret {note.fret}"

    if note.modifier != Technique.NONE:
        tag = note.modifier.name
        if note.technique_target is not None:
            tag += f"->{note.technique_target}"
        if note.bend_amount is not None:
            tag += f" +{note.bend_amount}steps"
        desc += f" [{tag}]"

    if note.arrival != Technique.NONE:
        desc += f" (arrival={note.arrival.name})"

    return desc


def _describe_hand_position(hp: HandPosition) -> str:
    if hp.finger is None:
        return "no finger needed (open string/muted)"
    finger_name = {1: "index", 2: "middle", 3: "ring", 4: "pinky"}.get(hp.finger, str(hp.finger))
    return f"finger {hp.finger} ({finger_name}), anchor fret {hp.anchor_fret}"


def format_fingering_report(moments: list[Moment], assignments: list[list[HandPosition]]) -> str:
    """Builds the report as a single string -- split out from
    `write_fingering_report` so it's independently testable/printable
    without touching disk."""
    lines = []
    for i, (moment, assignment) in enumerate(zip(moments, assignments), start=1):
        t = f"{moment.start_time:6.2f}s" if moment.start_time is not None else "  ??  "
        dur = f"{moment.duration:5.2f}s" if moment.duration is not None else " ?? "
        lines.append(f"Moment {i:>4} | col={moment.column:>5} | t={t} | dur={dur}")
        for hp in assignment:
            note = hp.note
            lines.append(f"    {note.string_name} string, {_describe_note(note)} -> {_describe_hand_position(hp)}")
        lines.append("")

    return "\n".join(lines)


def write_fingering_report(path: str, moments: list[Moment], assignments: list[list[HandPosition]]) -> None:
    """Writes `format_fingering_report`'s output to `path` (plain
    text, UTF-8)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(format_fingering_report(moments, assignments))


def next_report_path(reports_dir: str, stem: str) -> str:
    """
    Picks a filename for this run's report that won't clobber an
    earlier one for the same song: the first run for `stem` gets
    "<stem>_fingering.txt"; if that already exists, the next gets
    "<stem>_fingering_v_002.txt", then "_v_003.txt", and so on --
    the plain unsuffixed name always means "the first one," matching
    how the first version of anything usually isn't labeled "v1."
    Does NOT create the file itself, just decides where it should go
    (`write_fingering_report` does the actual writing) -- kept separate
    so this is independently testable without touching disk.
    """
    base_path = os.path.join(reports_dir, f"{stem}_fingering.txt")
    if not os.path.exists(base_path):
        return base_path

    version = 2
    while True:
        candidate = os.path.join(reports_dir, f"{stem}_fingering_v_{version:03d}.txt")
        if not os.path.exists(candidate):
            return candidate
        version += 1


if __name__ == "__main__":
    from src.parser import parse_tab
    from src.moments import group_into_moments
    from src.rhythm import assign_timing
    from src.optimizer import FingeringOptimizer

    with open("data/hotel_california_solo.txt", encoding="utf-8") as f:
        raw = f.read()
    lines = raw.split("\n")
    section = "\n".join(lines[5:11])  # small hand-checkable sample, matches other modules' __main__ demos

    notes = parse_tab(section)
    moments = assign_timing(group_into_moments(notes), bpm=75)
    solution = FingeringOptimizer().solve(moments)

    print(format_fingering_report(moments, solution))
