"""
parser.py

Converts plain-text guitar tabs (copied from Ultimate Guitar or similar)
into a list of structured Note objects that a fingering optimizer can
consume.

MODEL OF A TAB FILE
--------------------
A tab file is not one continuous grid. It's a series of "blocks" — groups
of 6 lines (one per string) that get reset every ~20-40 characters so the
tab fits on screen. Between blocks there's usually a blank line, a chord
name, a section label ("[Chorus]"), or lyrics.

So the pipeline here is:
    raw text -> split into blocks -> parse each block's 6 lines -> stitch
    the blocks back into one timeline by column offset.

"Time" = column index, per your constraint. No note-duration modeling.
"""

from dataclasses import dataclass
from enum import Enum
import re


# Top-to-bottom order strings are printed in for standard 6-string guitar,
# used ONLY as a fallback when a line has no string-name label (e.g. "e|").
STANDARD_TUNING_ORDER = ["e", "B", "G", "D", "A", "E"]

# Characters that can legitimately appear inside a tab line's content
# (i.e. everything after the "e|" label). Used to detect whether a line
# IS a tab line at all, vs. lyrics/section headers/chord names.
ALLOWED_CONTENT_CHARS = set("-0123456789hHpPxXbB/\\~()")

# Real guitars go up to ~24 frets. Used to catch source-formatting slips
# where two single-digit notes got typed with no separating dash.
MAX_REALISTIC_FRET = 24


class Technique(Enum):
    NONE = "none"
    HAMMER_ON = "hammer_on"
    PULL_OFF = "pull_off"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"
    BEND = "bend"
    VIBRATO = "vibrato"

# ^ means "bend fully up to the next note" — this is a special case of bend that doesn't have a target fret number. 
# It's not handled in the current parser, but could be added later if needed. For now, it will be ignored and treated as a normal note.

# Characters that describe the TRANSITION into the next note on the same
# string (i.e. they sit between two fret numbers and modify the second one).
ARRIVAL_CHARS = {
    "h": Technique.HAMMER_ON,
    "p": Technique.PULL_OFF,
    "/": Technique.SLIDE_UP,
    "\\": Technique.SLIDE_DOWN,
}

# Characters that decorate a note that was ALREADY played, rather than
# describing a transition to a new one. Vibrato has no target; bend does
# ("5b7" = bend fret 5 up until it sounds like fret 7 — this is still ONE
# physical note, not two).
MODIFIER_CHARS = {"~": Technique.VIBRATO}  # "b" is handled specially below


@dataclass
class Note:
    string_index: int          # 0 = top line as printed, 5 = bottom line
    string_name: str           # "e", "B", "G", "D", "A", "E"
    fret: int                  # fret number; -1 for a dead/muted note ("x")
    column: int                # absolute column across the WHOLE stitched song
    arrival: Technique = Technique.NONE    # how we got here from the prior note
    modifier: Technique = Technique.NONE   # decoration applied to this note itself
    bend_target: int | None = None         # only set when modifier == BEND

    def __repr__(self):
        base = f"Note(str={self.string_name}, fret={self.fret}, col={self.column}"
        if self.arrival != Technique.NONE:
            base += f", arrival={self.arrival.name}"
        if self.modifier != Technique.NONE:
            base += f", modifier={self.modifier.name}"
            if self.bend_target is not None:
                base += f"->{self.bend_target}"
        return base + ")"


def is_tab_line(line: str) -> bool:
    """
    Heuristic: is this line part of a tab grid, or something else (lyrics,
    section header, blank line, chord names)?

    We strip an optional leading string-name label like "e|" or "D|", strip
    a trailing "|", and check that at least 90% of what's left is made of
    characters that actually show up in tab notation. Lyrics and headers
    will fail this check hard (they're full of letters/spaces that aren't
    h/p/x/b).
    """
    stripped = line.strip()
    if len(stripped) < 4:
        return False

    content = re.sub(r'^[A-Ga-g](#|b)?[:|]?', '', stripped)
    content = content.rstrip('|')
    if not content:
        return False

    allowed = sum(1 for c in content if c in ALLOWED_CONTENT_CHARS)
    return (allowed / len(content)) >= 0.9


def group_into_blocks(lines: list[str]) -> list[list[str]]:
    """
    Walk the raw file line by line, collecting consecutive runs of
    is_tab_line() == True lines into blocks. Anything else (blank lines,
    lyrics, "[Verse 1]") acts as a separator between blocks.
    """
    blocks: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        if is_tab_line(line):
            current.append(line)
        else:
            if current:
                blocks.append(current)
                current = []
    if current:
        blocks.append(current)

    return blocks


def parse_line(content: str, string_index: int, string_name: str) -> list[Note]:
    """
    Parse a SINGLE string's line content (label and pipes already stripped)
    into a list of Notes, with columns relative to the start of this block
    (offset correction happens later in parse_tab).

    Two-pass approach:
      Pass 1 (tokenize): walk character by character. Multi-digit fret
      numbers get consumed as one token (so "12" doesn't become two frets
      "1" and "2"). "b" gets special-cased to also consume a following
      number as its bend target, since that number is NOT a separate note.
      Pass 2 (assemble): turn the token stream into Note objects, attaching
      arrival techniques to the note that follows them and modifiers to the
      note that precedes them.
    """
    tokens = []  # each is (column, kind, value) where kind in {"fret","arrival","modifier"}
    i = 0
    n = len(content)

    while i < n:
        c = content[i]

        if c.isdigit():
            start = i
            digits = c
            i += 1
            # Real guitars top out around 24 frets. Real tabs sometimes
            # have two separate single-digit notes typed with no dash
            # between them (a formatting slip in the source, common in
            # scraped/copy-pasted tabs). Without this check, "1197" reads
            # as fret 1197 — impossible, and silently wrong. So: consume
            # at most 2 digits, and if those 2 digits form a number over
            # MAX_REALISTIC_FRET, back off and treat it as two separate
            # single-digit notes instead.
            if i < n and content[i].isdigit():
                two_digit_value = int(digits + content[i])
                if two_digit_value <= MAX_REALISTIC_FRET:
                    digits += content[i]
                    i += 1
                else:
                    print(
                        f"WARNING: column {start}: '{digits}{content[i]}' "
                        f"(fret {two_digit_value}) exceeds a realistic fret "
                        f"number. Splitting into separate notes '{digits}' "
                        f"and '{content[i]}' — verify this against the "
                        f"original tab, the source formatting may be off."
                    )
            tokens.append((start, "fret", int(digits)))

        elif c in ("x", "X"):
            tokens.append((i, "fret", -1))  # dead/muted note, no real pitch
            i += 1

        elif c in ("b", "B"):
            start = i
            i += 1
            target = None
            if i < n and content[i].isdigit():
                digits = ""
                while i < n and content[i].isdigit():
                    digits += content[i]
                    i += 1
                target = int(digits)
            tokens.append((start, "modifier", (Technique.BEND, target)))

        elif c.lower() in ARRIVAL_CHARS:
            tokens.append((i, "arrival", ARRIVAL_CHARS[c.lower()]))
            i += 1

        elif c in MODIFIER_CHARS:
            tokens.append((i, "modifier", (MODIFIER_CHARS[c], None)))
            i += 1

        else:
            # dashes, stray spaces, parentheses (grace notes — not handled
            # yet, see limitations) all fall here and are skipped.
            i += 1

    # Pass 2: assemble tokens into Notes.
    notes: list[Note] = []
    pending_arrival = Technique.NONE
    idx = 0
    while idx < len(tokens):
        col, kind, val = tokens[idx]

        if kind == "fret":
            note = Note(
                string_index=string_index,
                string_name=string_name,
                fret=val,
                column=col,
                arrival=pending_arrival,
            )
            pending_arrival = Technique.NONE

            # If the very next token is a modifier, it decorates THIS note
            # (e.g. "7~" — vibrato on the note we just created), not a
            # future one. Attach it and consume it now.
            if idx + 1 < len(tokens) and tokens[idx + 1][1] == "modifier":
                technique, target = tokens[idx + 1][2]
                note.modifier = technique
                note.bend_target = target
                idx += 1

            notes.append(note)

        elif kind == "arrival":
            pending_arrival = val

        # kind == "modifier" with no preceding fret in this line is an
        # orphan token (malformed tab) — silently ignored.

        idx += 1

    return notes


def parse_block(block_lines: list[str]) -> tuple[list[Note], int]:
    """
    Parse one block (a run of consecutive tab lines, normally 6 of them —
    one per string). Returns (notes, block_width) where block_width is used
    by the caller to offset the NEXT block's columns so the timeline stays
    continuous across the whole song.
    """
    notes: list[Note] = []
    max_len = 0

    for i, raw_line in enumerate(block_lines):
        stripped = raw_line.strip()

        # Match the leading letter itself (e/B/G/D/A/E), regardless of
        # whether it's followed by "|", ":", or nothing at all. Real tabs
        # are inconsistent about the delimiter, and getting the STRING
        # NAME WRONG is a much worse bug than getting the delimiter
        # character wrong — it silently assigns notes to the wrong pitch.
        # Only fall back to position-based guessing when there's truly no
        # letter to read.
        label_match = re.match(r'^([A-Ga-g])(#|b)?', stripped)
        if label_match:
            string_name = label_match.group(1)
            rest = stripped[label_match.end():]
            if rest[:1] in (":", "|"):
                rest = rest[1:]
            content = rest
        else:
            # No label found — fall back to standard top-to-bottom order.
            # This will misbehave on 7-string or drop-tuned tabs with no
            # labels; flagged as a known limitation below.
            string_name = STANDARD_TUNING_ORDER[i] if i < len(STANDARD_TUNING_ORDER) else f"string{i}"
            content = stripped.lstrip('|')

        content = content.rstrip('|')

        max_len = max(max_len, len(content))
        notes.extend(parse_line(content, string_index=i, string_name=string_name))

    return notes, max_len


def parse_tab(raw_text: str) -> list[Note]:
    """
    Top-level entry point. Splits the raw file into blocks, parses each,
    and stitches column indices together so columns are unique across the
    ENTIRE song, not just within one block.
    """
    lines = raw_text.split("\n")
    blocks = group_into_blocks(lines)

    all_notes: list[Note] = []
    column_offset = 0

    for block in blocks:
        block_notes, block_width = parse_block(block)
        for note in block_notes:
            note.column += column_offset
        all_notes.extend(block_notes)
        column_offset += block_width

    return all_notes


if __name__ == "__main__":
    # Small hand-checkable sample covering: plain notes, multi-digit frets,
    # hammer-on, pull-off, slide, and a bend with target. Trace through
    # parse_line by hand once — it's the only way to trust this file.
    sample = """
e|--5h7---7p5--10-12/14--|
B|------------------------|
G|------------------------|
D|------------------------|
A|------------------------|
E|------------------------|

e|--0----0-------0--|
B|----1------1-------|
G|-------0------------|
D|--------------------|
A|--------------------|
E|--------------------|
""".strip("\n")

    notes = parse_tab(sample)
    notes.sort(key=lambda note: note.column)
    for note in notes:
        print(note)