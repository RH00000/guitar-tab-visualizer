"""
parser.py

Converts plain-text guitar tabs into a list of structured Note objects 
that a fingering optimizer can consume.

MODEL OF A TAB FILE
--------------------
A tab file is not one continuous grid. It's a series of "blocks": groups
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
ALLOWED_CONTENT_CHARS = set("-0123456789hHpPxXbBsS/\\~()^|")

# Real guitars go up to ~24 frets. Used to catch source-formatting slips
# where two single-digit notes got typed with no separating dash.
MAX_REALISTIC_FRET = 24

# how far to run an indefinite slide when no target fret is given 
# intentionally kept as a constant to make it easy to change in one place if a different default is desired
INDEFINITE_SLIDE_RUN_FRETS = 8


class Technique(Enum):
    NONE = "none"   
    HAMMER_ON = "hammer_on"
    PULL_OFF = "pull_off"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"
    SLIDE = "slide"            # "s" — direction unknown from the character alone; resolved once the target fret is known
    BEND = "bend"
    VIBRATO = "vibrato"
    RELEASE = "release"       # "r" — bend released back down
    PRE_BEND = "pre_bend"      # "pb" — string bent BEFORE being picked
    REBEND = "rebend"          # "rb" — after a bend+release, bent again


# Characters that describe the TRANSITION into the next note on the same
# string (i.e. they sit between two fret numbers and modify the second one).
ARRIVAL_CHARS = {
    "h": Technique.HAMMER_ON,
    "p": Technique.PULL_OFF,
    "/": Technique.SLIDE_UP,
    "\\": Technique.SLIDE_DOWN,
    "s": Technique.SLIDE,  # direction unknown yet — resolved in the assembly pass
}

# Characters that decorate a note that was ALREADY played, rather than
# describing a transition to a new one. Vibrato has no target.
MODIFIER_CHARS = {"~": Technique.VIBRATO}  # "b" is handled specially below


@dataclass
class Note:
    string_index: int          # 0 = top line as printed, 5 = bottom line
    string_name: str           # "e", "B", "G", "D", "A", "E"
    fret: int                  # fret number; -1 for a dead/muted note ("x")
    column: int                # absolute column across the WHOLE stitched song
    arrival: Technique = Technique.NONE    # how we got here from the prior note
    modifier: Technique = Technique.NONE   # decoration applied to this note itself
    technique_target: int | None = None      # target fret/pitch for BEND, RELEASE, or PRE_BEND — None if not specified
    bend_amount: float | None = None         # step amount from "(1/2)"/"(2)" style notation — a DIFFERENT convention than technique_target, some tabs use one, some the other

    def __repr__(self):
        base = f"Note(str={self.string_name}, fret={self.fret}, col={self.column}"
        if self.arrival != Technique.NONE:
            base += f", arrival={self.arrival.name}"
        if self.modifier != Technique.NONE:
            base += f", modifier={self.modifier.name}"
            if self.technique_target is not None:
                base += f"->{self.technique_target}"
            if self.bend_amount is not None:
                base += f"(+{self.bend_amount} steps)"
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


def _consume_bend_qualifier(content: str, i: int, n: int) -> tuple[int | None, float | None, int]:
    """
    Shared by every bend-family technique ('b', 'pb', 'rb', 'r', '^').
    Different tabs qualify a bend two DIFFERENT, incompatible ways, and
    this project has now seen real files using both:
      - target-fret notation: "5b7" = bend 5 up to sound like fret 7
      - step-amount notation: "15b(2)" = bend 15 up by 2 whole steps
    Without handling the second form explicitly, the tokenizer read
    "(2)" as filler + a brand new note at fret 2 — not just lost
    information, an ACTIVELY FABRICATED note that was never played.
    Returns (target_or_None, amount_or_None, new_i) — at most one of
    target/amount will be set, matching whichever notation was used.
    """
    if i < n and content[i].isdigit():
        digits = ""
        while i < n and content[i].isdigit():
            digits += content[i]
            i += 1
        return int(digits), None, i

    if i < n and content[i] == "(":
        close = content.find(")", i)
        if close == -1:
            # Malformed — no closing paren found. Don't guess; just
            # consume the "(" itself as filler and move on.
            return None, None, i + 1
        inner = content[i + 1:close]
        i = close + 1
        if "/" in inner:
            num, _, denom = inner.partition("/")
            try:
                return None, float(num) / float(denom), i
            except (ValueError, ZeroDivisionError):
                return None, None, i
        try:
            return None, float(inner), i
        except ValueError:
            return None, None, i

    return None, None, i


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
            # scraped/copy-pasted tabs).
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
            target, amount, i = _consume_bend_qualifier(content, i, n)
            tokens.append((start, "modifier", (Technique.BEND, target, amount)))

        elif c in ("p", "P") and i + 1 < n and content[i + 1] in ("b", "B"):
            # "pb" = pre-bend: the string is ALREADY bent before it's
            # picked. This has to be checked before the generic ARRIVAL
            # "p" (pull-off) case below, or "pb10" gets misread as
            # "pull-off" + a dangling, unattached bend-to-10 that then
            # silently gets dropped and the pull-off tag wrongly carries
            # forward onto some later, unrelated note.
            start = i
            i += 2  # consume "pb" together
            target, amount, i = _consume_bend_qualifier(content, i, n)
            tokens.append((start, "modifier", (Technique.PRE_BEND, target, amount)))

        elif c.lower() in ARRIVAL_CHARS:
            tokens.append((i, "arrival", ARRIVAL_CHARS[c.lower()]))
            i += 1

        elif c in ("r", "R") and i + 1 < n and content[i + 1] in ("b", "B"):
            # "rb" = rebend: after an initial bend-and-release, bent a
            # SECOND time. NOT the same technique as pre-bend ("pb") —
            # pre-bend happens before the note is even picked, rebend is
            # a second bend later in the same note's life — but parsed
            # the same structural way, checked before bare "r" for the
            # same reason "pb" is checked before bare "p".
            start = i
            i += 2
            target, amount, i = _consume_bend_qualifier(content, i, n)
            tokens.append((start, "modifier", (Technique.REBEND, target, amount)))

        elif c in ("r", "R"):
            # Bare "r" = release (bend let back down). Without this,
            # "11r9" was being read as fret 11, THEN a totally separate
            # new note at fret 9 — wrong; the "9" is the release's
            # target pitch, not an independent note.
            start = i
            i += 1
            target, amount, i = _consume_bend_qualifier(content, i, n)
            tokens.append((start, "modifier", (Technique.RELEASE, target, amount)))

        elif c == "^":
            # "^" = full bend (typically a whole-step bend with no
            # explicit target fret given, unlike "b7" which names one).
            # Same Technique.BEND as "b", just no digit-target to consume.
            start = i
            target, amount, i = _consume_bend_qualifier(content, i + 1, n)
            tokens.append((start, "modifier", (Technique.BEND, target, amount)))

        elif c in MODIFIER_CHARS:
            tokens.append((i, "modifier", (MODIFIER_CHARS[c], None, None)))
            i += 1

        else:
            # dashes, stray spaces, parentheses (grace notes — not handled
            # yet, see limitations) all fall here and are skipped.
            i += 1

    # Pass 2: assemble tokens into Notes.
    notes: list[Note] = []
    pending_arrival = Technique.NONE
    pending_arrival_col = None  # column of the arrival CHARACTER itself, not the note it'll attach to
    idx = 0
    while idx < len(tokens):
        col, kind, val = tokens[idx]

        if kind == "fret":
            arrival = pending_arrival
            if arrival == Technique.SLIDE and notes:
                # "s" doesn't say which direction — figure it out by
                # comparing to the note we just came from on this SAME
                # string (notes[-1], since parse_line handles one
                # string at a time).
                arrival = Technique.SLIDE_UP if val >= notes[-1].fret else Technique.SLIDE_DOWN

            note = Note(
                string_index=string_index,
                string_name=string_name,
                fret=val,
                column=col,
                arrival=arrival,
            )
            pending_arrival = Technique.NONE

            # If the very next token is a modifier, it decorates THIS note
            # (e.g. "7~" — vibrato on the note we just created), not a
            # future one. Attach it and consume it now.
            if idx + 1 < len(tokens) and tokens[idx + 1][1] == "modifier":
                technique, target, amount = tokens[idx + 1][2]
                note.modifier = technique
                note.technique_target = target
                note.bend_amount = amount
                idx += 1

            notes.append(note)

        elif kind == "arrival":
            pending_arrival = val
            pending_arrival_col = col

        # kind == "modifier" with no preceding fret in this line is an
        # orphan token (malformed tab) — silently ignored.

        idx += 1

    # An arrival token that's STILL pending after the loop ends means
    # it was never followed by a fret token on this line — the
    # "indefinite slide" case ("10/" with nothing after the "/",
    # meaning "slide off, direction implied, exact ending fret
    # unspecified"). Without this, the "/" set pending_arrival and then
    # nothing ever consumed it: no destination Note was created at all,
    # silently dropping a real, audible technique from the song.
    if pending_arrival in (Technique.SLIDE_UP, Technique.SLIDE_DOWN) and notes:
        origin = notes[-1]
        direction = 1 if pending_arrival == Technique.SLIDE_UP else -1
        raw_target = origin.fret + direction * INDEFINITE_SLIDE_RUN_FRETS
        target_fret = max(0, min(MAX_REALISTIC_FRET, raw_target))

        synthesized = Note(
            string_index=string_index,
            string_name=string_name,
            # Column just after the arrival character itself, so it
            # sorts immediately after the origin note but doesn't
            # collide with any real token column on this line.
            column=pending_arrival_col + 1,
            fret=target_fret,
            # This is the field src/optimizer.py's slide-continuity
            # cost reads to require the SAME finger across the slide —
            # setting it to anything else (or leaving it NONE) would
            # make the optimizer silently never apply that rule here.
            arrival=pending_arrival,
        )
        notes.append(synthesized)

        direction_word = "up" if pending_arrival == Technique.SLIDE_UP else "down"
        print(
            f"WARNING: column {pending_arrival_col}: indefinite slide "
            f"(fret {origin.fret} sliding {direction_word} with no target "
            f"fret given) — inventing a destination {INDEFINITE_SLIDE_RUN_FRETS} "
            f"frets {direction_word} at fret {target_fret}. This is an "
            f"approximation of an intentionally unmeasured technique, not "
            f"a real transcribed value — verify against the original tab."
        )
        if target_fret != raw_target:
            print(
                f"WARNING: column {pending_arrival_col}: indefinite slide's "
                f"invented destination (fret {raw_target}) fell outside the "
                f"realistic fretboard range (0-{MAX_REALISTIC_FRET}) — "
                f"clamped to fret {target_fret} instead. The actual slide "
                f"target is now different from a literal "
                f"{INDEFINITE_SLIDE_RUN_FRETS}-fret run — verify against the "
                f"original tab."
            )
    elif pending_arrival == Technique.SLIDE:
        # Bare "s" with no target fret: not just missing a destination,
        # missing the DIRECTION too, and there's no following fret to
        # compare against (the usual way SLIDE gets resolved into
        # SLIDE_UP/SLIDE_DOWN above). Fabricating a direction with no
        # basis would be worse than not fabricating a note at all, so
        # this is left out of the synthesis entirely — same as current
        # behavior, just now explicitly flagged instead of silently
        # dropped.
        print(
            f"WARNING: column {pending_arrival_col}: indefinite slide with "
            f"unknown direction ('s' with no target fret to compare "
            f"against) — cannot infer a direction, so no destination note "
            f"was synthesized. Verify against the original tab."
        )

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
        # whether it's followed by "|", ":", or nothing at all.
        label_match = re.match(r'^([A-Ga-g])(#|b)?', stripped)
        if label_match:
            string_name = label_match.group(1)
            rest = stripped[label_match.end():]
            if rest[:1] in (":", "|"):
                rest = rest[1:]
            content = rest
        else:
            # No label found . fall back to standard top-to-bottom order.
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
    entire song, not just within one block.
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

