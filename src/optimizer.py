"""
optimizer.py

Chooses one hand-fingering assignment per Moment for the whole song,
minimizing total physical effort. This is the resume-worthy piece: a
dynamic program over HAND SHAPES, not over individual notes.

THE PHYSICAL MODEL
--------------------
A guitarist's fretting hand sits at some ANCHOR FRET at any instant —
roughly, where the index finger rests. From that anchor, each finger
naturally covers one fret: index=anchor, middle=anchor+1, ring=
anchor+2, pinky=anchor+3. So playing a note at fret F with finger f
implies anchor = F - (f-1); that's exactly `HandPosition.anchor_fret`.

A single note has up to 4 candidate fingerings (which finger plays it).
A moment with multiple simultaneous notes (a dyad) is a JOINT decision:
the fingers used must all be different, AND they must be reachable
from close to the SAME anchor at once — you only have one hand. That's
why the DP's nodes are whole-moment assignments, not per-note choices
(see METHODS_AND_INTERFACES.md — this is the actual fix over a naive
"decide each note independently" approach, which can't represent "you
can't use your ring finger twice at once").

THREE COSTS, ALL IN PHYSICAL OR PHYSICAL-ADJACENT UNITS
----------------------------------------------------------
- STRETCH cost: how spread apart a moment's OWN anchors are (a chord's
  internal difficulty), priced by `stretch_weight`.
- SHIFT cost: how far the hand's anchor has to travel from the
  PREVIOUS moment's shape to this one, priced by `shift_weight` and
  discounted when there's more time available to make the move.
- FINGER cost: which finger got used, independent of stretch/shift.
  Found necessary empirically, not designed in from the start — see
  the note in `__init__` and `_finger_cost` below. Without it, finger
  choice is a free variable: because `anchor = fret - (finger - 1)`,
  the DP can redefine a note's anchor just by picking a different
  finger for it, and it will do so purely to make `shift_cost` cheaper
  (e.g. reaching a note with the pinky instead of the index, just to
  keep the anchor parked near a neighboring moment) — a "free" trick
  that has nothing to do with how comfortable that finger choice
  actually is. `finger_weight` prices the finger itself, and
  `technique_weight` adds an extra charge for using the pinky
  specifically on a bend or vibrato, where it's rare and awkward in
  real playing (ring finger should be doing that work instead).

STRETCH, SHIFT, and FINGER's baseline ALL use PHYSICAL fret distance,
not raw fret-count: frets narrow toward the body under standard
equal-tempered spacing (each fret is 2**(-1/12) ≈ 0.9439 the WIDTH of
the fret before it — the guitar's 12th fret sits at exactly half the
scale length). A 2-fret jump at fret 3 is a real reach; the same
2-fret jump at fret 15 is nearly nothing. Treating both as "2" would
bias the optimizer away from upper-fret passages that are actually
easy — so every distance calculation here goes through
`physical_fret_position` first, never raw fret subtraction.

FINGER cost was originally a flat per-finger-step constant instead
(pinky = 3x the cost of middle, everywhere on the neck, regardless of
position) — that turned out to be a real bug, not just a rougher
approximation: a flat cost never shrinks, but SHIFT's physical
distance does shrink toward zero up the neck by construction. That
mismatch meant there was always some fret position above which
relocating the whole hand got priced as cheaper than just switching
fingers while staying put — not because that's realistic, but purely
because the two costs lived on different scales. Pricing FINGER's
baseline as a physical reach too (see `_finger_cost`) fixes this
structurally: all three costs now shrink together at the same rate up
the neck, so trading a finger switch for a hand relocation is decided
by genuinely available `rest_time` (correct), not by raw fretboard
position (a units-mismatch artifact). The one deliberate exception is
`technique_weight` (the extra pinky-on-bend/vibrato charge): that's a
flat constant on purpose, because it prices a technique-difficulty
preference, not a spatial one — awkwardness of pinky string-bending
doesn't get easier just because you're higher up the neck.

A CORRECTED INTERPRETATION OF "REST TIME" (found while building this,
not assumed away)
------------------------------------------------------------------------
METHODS_AND_INTERFACES.md describes transition_cost's rest_time as
"previous moment's end -> next moment's start." Taken literally, that
is ALWAYS ZERO under this project's rhythm.py: duration is defined as
"hold until the next moment's onset," so a moment's end and the next
moment's start are the same instant by construction — there is no
literal silence to measure. The actual intent is "how much time is
available to move the hand before the next note must sound," which IS
a real, nonzero, useful quantity: the INTER-ONSET interval, i.e. the
previous moment's `duration` (= next.start_time - this.start_time).
That's what `solve()` actually passes as rest_time. Using the literal
"end to start" definition would make the entire discount feature a
dead code path that never discounts anything — worth catching before
building it, not after.

THE DP ITSELF
--------------
- Layer i = moment i (not note i — a chord gets ONE joint decision).
- Node = one full joint assignment for that moment (one HandPosition
  per note), from `candidate_assignments`.
- Edge (i-1 -> i) cost = `transition_cost`, using the inter-onset
  interval between the two moments as rest_time.
- `solve` does the standard forward pass (accumulate the cheapest way
  to reach each node, remembering which predecessor achieved it) then
  a backward trace from the cheapest final node — the same shape as
  textbook edit-distance / Viterbi DP, just with "hand shape" as the
  state instead of "row/column" or "hidden state."
"""

import itertools
import math
from dataclasses import dataclass

from src.moments import Moment
from src.parser import Note, Technique

# Bend-family techniques: all of these involve the fretting finger
# actively pushing/holding the string under tension, which is the
# physical motion that's awkward with the pinky.
#
# Note: as of this writing, parser.py's parse_line only ever attaches
# ONE modifier token to a Note (whichever immediately follows its fret
# token) -- so a real "bend then release" ("11b13r11") ends up with
# modifier=BEND only; the RELEASE token is silently orphaned and
# dropped before it ever reaches a Note. RELEASE/REBEND are included
# here for correctness and for the standalone case ("11r9" with no
# preceding "b"), but won't fire on the more common compound
# bend-and-release pattern until that parser-level bug is fixed
# separately. Not this file's bug to fix, but worth knowing why
# RELEASE/REBEND may look like dead weight in testing until it is.
BEND_FAMILY = (Technique.BEND, Technique.PRE_BEND, Technique.REBEND, Technique.RELEASE)

# Two separate magnitude-based cost tables, not one shared rank table.
# A flat rank (0/1/2/3) forces equal spacing between every finger
# step, which doesn't match real technique difficulty for either
# technique, and the two techniques don't even share the same shape of
# difficulty:
#   - BEND_FINGER_COST: bending is a sustained strength/control
#     problem (holding a string under tension in tune), and there's a
#     real gradient to it -- ring is easiest (the two fingers behind it
#     on the neck can brace it), then middle, then index (weakest of
#     the three without help behind it), with pinky a much BIGGER jump
#     than the even spacing between the others (weakest finger
#     overall, genuinely awkward to hold string tension with, not just
#     "one more rank down").
#   - VIBRATO_FINGER_COST: vibrato is an oscillation, not a sustained
#     push, so it's close to finger-agnostic for index/middle/ring --
#     the only real penalty is pinky.
# Both deliberately do NOT match the general `_finger_cost` baseline's
# index-is-always-cheapest assumption -- that baseline is about
# physical REACH from the anchor (index reaches furthest for free
# because anchor is defined by it), which has nothing to do with which
# finger has the STRENGTH/control to bend or oscillate a string well.
# A lone bent note with plenty of rest_time around it should still
# prefer ring over index even though index "reaches for free," because
# bending isn't a reach problem, it's a strength problem.
# These are starting values the project owner will tune by eye, not
# measured constants.
BEND_FINGER_COST = {3: 0.0, 2: 0.3, 1: 0.6, 4: 1.2}  # ring, middle, index, pinky
VIBRATO_FINGER_COST = {3: 0.0, 2: 0.0, 1: 0.0, 4: 1.0}  # only pinky is penalized

# Sliding lives on Note.arrival (how you got to this note), not
# Note.modifier (how the note is decorated once you're on it) --
# checking the wrong field would silently never match, same class of
# bug as an earlier enum/string mismatch in this file. Bare
# Technique.SLIDE is included alongside SLIDE_UP/SLIDE_DOWN because
# parser.py's parse_line only resolves "s" into a direction when
# there's a prior note on the same string to compare against -- the
# first note on a string with a leading "s" has no prior note yet and
# can reach here still unresolved. Like vibrato, sliding is close to
# finger-agnostic except for a pinky penalty.
SLIDE_TECHNIQUES = (Technique.SLIDE_UP, Technique.SLIDE_DOWN, Technique.SLIDE)
SLIDE_FINGER_COST = {3: 0.0, 2: 0.0, 1: 0.0, 4: 1.0}

# Hammer-ons/pull-offs, unlike slides, need a SECOND finger already (or
# about to be) down on the same string -- see _legato_conflict_cost and
# _pulloff_prepositioning_cost below.
HAMMER_PULL_TECHNIQUES = (Technique.HAMMER_ON, Technique.PULL_OFF)

FINGERS = (1, 2, 3, 4)  # 1=index, 2=middle, 3=ring, 4=pinky


def physical_fret_position(fret: float) -> float:
    """
    Physical distance from the nut to `fret`, as a fraction of scale
    length (0.0 = nut / open string, 0.5 = the 12th fret, by the
    standard equal-tempered guitar formula). Frets get narrower toward
    the body: fret n's WIDTH is 2**(-1/12) (~0.9439) the width of fret
    n-1, not a constant width — this is why raw fret-count is the
    wrong unit for "how far is this stretch/shift, really."
    """
    return 1.0 - 2.0 ** (-fret / 12.0)


def physical_distance(fret_a: float, fret_b: float) -> float:
    """Physical distance between two frets, in the same scale-length units."""
    return abs(physical_fret_position(fret_a) - physical_fret_position(fret_b))


@dataclass
class HandPosition:
    note: Note
    finger: int | None        # 1-4, or None for open strings / muted notes
    anchor_fret: int | None   # fret - (finger - 1); None when finger is None


class FingeringOptimizer:
    """
    Config:
      stretch_weight    -- cost per unit of physical spread WITHIN one
                            moment's own hand shape (chord difficulty).
      shift_weight       -- cost per unit of physical distance the hand
                            anchor moves BETWEEN moments (position-change
                            difficulty), discounted by available rest time.
      max_stretch_frets  -- how many frets of anchor spread one hand can
                            cover in a single moment before a joint
                            assignment is rejected as physically infeasible.
      finger_weight      -- cost per PHYSICAL unit of reach a finger makes
                            FROM the moment's anchor to reach its actual
                            fret (0 for index, since anchor is defined as
                            index's own fret; growing for middle/ring/
                            pinky by the real, position-dependent physical
                            distance -- same units and same
                            `physical_fret_position` law as stretch/shift,
                            not a flat per-finger-step count). Exists to
                            stop finger choice from being a free variable
                            the DP can use to game shift_cost -- confirmed
                            empirically: without this, an isolated
                            high-fret note would get assigned to the
                            pinky just to keep the anchor parked near a
                            neighboring moment, not because the pinky was
                            actually the natural choice there.

                            AN EARLIER VERSION OF THIS PRICED IT AS A FLAT
                            CONSTANT PER FINGER-STEP (e.g. pinky = 3 *
                            finger_weight, unconditionally) -- that was
                            wrong, not just imprecise: a flat constant
                            never shrinks, but shift_cost's physical
                            distance DOES shrink toward zero up the neck
                            by construction (frets narrow toward the
                            body). That mismatch meant there was always
                            SOME fret position above which "relocate the
                            whole hand with the index finger" was priced
                            as cheaper than "stay anchored, use the ring
                            finger" -- not because that's realistic, but
                            purely because one cost was flat and the
                            other wasn't. No single flat constant could
                            fix this: lowering it to stop that just
                            reintroduced the original pinky-hugging bug
                            at LOW frets instead (confirmed: a sweep
                            across finger_weight in [0.018, 0.02, 0.022]
                            only moved the crossover fret up or down the
                            neck, it never removed it). Pricing the
                            reach in the SAME physical units as
                            shift_cost fixes this structurally: both
                            costs now shrink together at the same rate
                            up the neck, so whether relocating beats
                            finger-switching is driven by genuinely
                            available `rest_time` (correct -- a real
                            player WOULD relocate given enough time) not
                            by raw fretboard position (a units-mismatch
                            artifact).
      technique_weight   -- multiplier applied to the per-technique,
                            per-finger MAGNITUDE cost tables
                            (BEND_FINGER_COST, VIBRATO_FINGER_COST,
                            SLIDE_FINGER_COST, module level) for
                            bending, vibrato, and sliding respectively
                            -- see those tables' own docstrings for why
                            bending gets a real gradient (ring best,
                            pinky a much bigger jump than the rest)
                            while vibrato/slide are close to
                            finger-agnostic except for pinky.
      min_shift_discount -- floor on the rest_time discount applied to
                            shift_cost (see `transition_cost`).
                            rhythm.py's timing is a heuristic, not a
                            real transcription, so a generous or
                            overestimated rest_time can push the
                            discount toward zero, making the model
                            treat a hand relocation as nearly free even
                            when the real available time was
                            overestimated. The floor caps how much a
                            bad rest_time estimate can distort the cost
                            -- a defensive hedge against known-
                            approximate timing data, not a claim that
                            hand movement has some true minimum
                            physical cost.
      slide_continuity_weight -- cost when a slide (`Note.arrival` in
                            SLIDE_TECHNIQUES) is assigned a DIFFERENT
                            finger than whatever was already on that
                            string a moment ago (see
                            `_slide_continuity_cost`). A slide is one
                            continuous motion of a single finger along
                            the string -- you cannot switch fingers
                            mid-slide, so this is close to a hard
                            constraint rather than a soft preference,
                            priced high enough that the DP will only
                            ever pay it when every other option is
                            worse (e.g. the origin note truly can't be
                            reached at all, a data problem upstream).
      legato_conflict_weight -- cost when a hammer-on/pull-off
                            (`Note.arrival` in HAMMER_PULL_TECHNIQUES)
                            is assigned the SAME finger as whatever was
                            already on that string a moment ago (see
                            `_legato_conflict_cost`) -- the opposite
                            requirement from sliding: hammering onto or
                            pulling off a fret needs a SECOND finger,
                            since the first one is what's already
                            sounding the origin note.
      pulloff_prepositioning_weight -- cost, scaled by how far past one
                            hand's reach the origin and destination
                            frets are, for a PULL_OFF specifically (not
                            HAMMER_ON) whose two frets aren't within a
                            single hand's real physical stretch of each
                            other (see `_pulloff_prepositioning_cost`).
                            A pull-off needs the destination finger
                            already resting on the string before the
                            origin note releases -- both fingers
                            coexist on the fretboard for an instant,
                            like a phantom two-note chord, even though
                            only one note sounds at a time. Hammer-on
                            has no equivalent requirement, since the
                            hammering finger can arrive fresh.
    """

    def __init__(
        self,
        stretch_weight: float = 1.0,
        shift_weight: float = 1.0,
        max_stretch_frets: int = 4,
        finger_weight: float = 0.5,
        technique_weight: float = 0.08,
        min_shift_discount: float = 0.35,
        slide_continuity_weight: float = 1.5,
        legato_conflict_weight: float = 1.5,
        pulloff_prepositioning_weight: float = 3.0,
    ):
        self.stretch_weight = stretch_weight
        self.shift_weight = shift_weight
        self.max_stretch_frets = max_stretch_frets
        self.finger_weight = finger_weight
        self.technique_weight = technique_weight
        self.min_shift_discount = min_shift_discount
        self.slide_continuity_weight = slide_continuity_weight
        self.legato_conflict_weight = legato_conflict_weight
        self.pulloff_prepositioning_weight = pulloff_prepositioning_weight

    def _note_candidates(self, note: Note) -> list[HandPosition]:
        """
        Every physically-possible (finger, anchor) option for ONE note,
        considered in isolation (no other notes in its moment yet).

        fret <= 0 covers both open strings (0, genuinely need no finger)
        and dead/muted notes (-1, "x" in the tab) -- a stated
        simplification: a muted hit is usually played by whichever
        finger is already resting nearby, so it isn't modeled as
        claiming a dedicated finger slot here.
        """
        if note.fret <= 0:
            return [HandPosition(note=note, finger=None, anchor_fret=None)]

        options = []
        for finger in FINGERS:
            anchor = note.fret - (finger - 1)
            if anchor < 0:
                continue  # this finger can't reach this fret from any real hand position
            # anchor == 0 is a real, valid hand position (right at the
            # nut) -- e.g. fret 1 played with the middle finger, index
            # resting near fret 0. It's distinct from an open string
            # (fret 0 itself, handled entirely above and never reaches
            # here): this is a FRETTED note, just played from the
            # lowest possible position. Excluding it was the original
            # bug -- two notes at the same low fret (a real pattern:
            # Master of Puppets' gallop riff repeats fret 1 on two
            # adjacent strings) would otherwise both be forced onto
            # finger 1 with no alternative, making them un-fingerable
            # as a pair even though a real player just uses two
            # different fingers close to the nut.
            options.append(HandPosition(note=note, finger=finger, anchor_fret=anchor))
        return options

    def candidate_assignments(self, moment: Moment) -> list[list[HandPosition]]:
        """
        Every physically valid way to assign fingers to ALL notes in
        `moment` at once. For a single-note moment that's just each
        note's own 1-4 options, unwrapped into single-element lists.
        For a multi-note moment (dyad, or occasionally more — this
        generalizes to any note count, though the project's target
        material is single notes/dyads), this takes the Cartesian
        product of each note's own options and keeps only combinations
        that are physically playable by one hand:
          1. a finger is only reused across two notes in the same
             moment when it's a genuine barre (same finger, same fret,
             different strings) -- reusing a finger across two
             DIFFERENT frets at once is rejected as physically
             impossible
          2. an open-string note (fret 0) never gets assigned a finger
             at all (automatic here — `_note_candidates` never offers
             one for fret <= 0, nothing to filter out after the fact)
          3. every note's implied anchor is within `max_stretch_frets`
             of every other note's implied anchor in the same combo —
             representing one hand's real physical reach, measured in
             PHYSICAL fret distance (see module docstring), not raw
             fret-count.

        A moment with more DISTINCT fretted positions than 4 (found in
        real data — e.g. a dense strummed chord, not a solo dyad)
        can't satisfy rule 1 at all, by pigeonhole: there just aren't
        enough fingers, independent of stretch. That's genuinely out
        of this project's scope and, on a solo tab, almost certainly a
        moments.py grouping artifact rather than a real chord. Handled
        the same way as the stretch fallback below: don't crash the
        whole song, use a placeholder, warn loudly.

        NOTE ON RAW NOTE COUNT vs. DISTINCT FRET COUNT (a real bug,
        found and fixed): an earlier version of this check rejected any
        moment with more than 4 fretted NOTES, full stop -- that's
        wrong for a real full BARRE CHORD, which routinely sounds 5-6
        strings with only 3-4 DISTINCT fret values (the barred fret
        covers several strings under one finger, by rule 1's own barre
        exception above; only the few "shape" notes on top of the barre
        need their own separate fingers). Rejecting on raw note count
        was throwing away exactly the cases the barre exception exists
        to solve -- a 6-note barre chord would get the crude
        every-note-gets-index placeholder below instead of ever
        attempting the real search, even when a genuine, physically
        correct barre-plus-3-fingers combination existed. The check
        now runs AFTER the actual search (using `finger_legal`, not a
        pre-count guess) so it only fires when NO legal combination
        exists at all, barre included -- true pigeonhole impossibility,
        not an approximation of it.
        """
        per_note_options = [self._note_candidates(n) for n in moment.notes]

        # `max_stretch_frets` is a fret-COUNT knob, but feasibility has
        # to be judged in physical units. Calibrate it against the
        # WIDEST possible spacing -- a span starting at the nut (fret
        # 0) -- since that's the hardest-case stretch for a given
        # fret-count. The payoff: a stretch of MORE than
        # `max_stretch_frets` frets higher up the neck, where spacing
        # is physically tighter, can still pass this check (correctly
        # easier), while the same fret-count near the nut is right at
        # the limit (correctly the hard case).
        max_physical_stretch = physical_distance(0, self.max_stretch_frets - 1)

        finger_legal: list[tuple[HandPosition, ...]] = []
        reachable: list[list[HandPosition]] = []

        for combo in itertools.product(*per_note_options):
            # A finger can be reused across notes in the same moment
            # ONLY as a genuine barre: the same finger pressing the
            # SAME fret on multiple strings at once (e.g. index finger
            # flat across fret 7 on both the A and low-E strings) is
            # exactly how real players cover this, not two separate
            # fingers each fighting for their own anchor. Reusing a
            # finger across two DIFFERENT frets is impossible (one
            # finger can't be in two places at once) and stays
            # rejected.
            finger_frets: dict[int, int] = {}
            barre_conflict = False
            for hp in combo:
                if hp.finger is None:
                    continue
                if hp.finger in finger_frets and finger_frets[hp.finger] != hp.note.fret:
                    barre_conflict = True
                    break
                finger_frets[hp.finger] = hp.note.fret
            if barre_conflict:
                continue

            finger_legal.append(combo)

            anchors = [hp.anchor_fret for hp in combo if hp.anchor_fret is not None]
            spread = physical_distance(max(anchors), min(anchors)) if len(anchors) >= 2 else 0.0
            if spread <= max_physical_stretch:
                reachable.append(list(combo))

        if not finger_legal:
            # Truly pigeonhole-impossible: even allowing every barre the
            # rules above permit, more distinct frets need covering than
            # there are fingers to cover them with. Confirmed via the
            # actual search, not guessed from a raw note count (see the
            # docstring above) -- e.g. 5+ genuinely different simultaneous
            # fret positions, no barre can rescue that.
            fretted_count = sum(1 for n in moment.notes if n.fret > 0)
            placeholder = [
                HandPosition(note=n, finger=1, anchor_fret=n.fret) if n.fret > 0
                else HandPosition(note=n, finger=None, anchor_fret=None)
                for n in moment.notes
            ]
            print(
                f"WARNING: column {moment.column}: {fretted_count} simultaneously "
                f"fretted notes across too many distinct frets for 4 fingers, even "
                f"allowing barres -- not representable by one hand at all. Using a "
                f"placeholder fingering so the rest of the song can still be "
                f"solved. Verify against the original tab -- moments.py's "
                f"grouping tolerance may have merged notes that aren't really "
                f"simultaneous."
            )
            return [placeholder]

        if not reachable:
            # Every finger-legal combo needs more of a stretch than one
            # hand can comfortably cover. Almost certainly either (a) a
            # moments.py grouping artifact -- two notes that were never
            # really simultaneous, just close enough in column to pass
            # `tolerance` -- or (b) a genuinely wide chord voicing, out
            # of this project's scope (see PLAN.md exclusions). Don't
            # crash the whole song over one bad moment: fall back to the
            # least-stretched option anyway, and say so loudly, same
            # "tell, don't hide" approach as the rest of this project.
            best = min(
                finger_legal,
                key=lambda combo: (
                    max((hp.anchor_fret for hp in combo if hp.anchor_fret is not None), default=0)
                    - min((hp.anchor_fret for hp in combo if hp.anchor_fret is not None), default=0)
                ),
            )
            notes_desc = [(n.string_name, n.fret) for n in moment.notes]
            print(
                f"WARNING: column {moment.column}: no finger combination for "
                f"this moment fits within one hand span (max_stretch_frets="
                f"{self.max_stretch_frets}). Notes involved: {notes_desc}. "
                f"Falling back to the least-stretched option anyway -- verify "
                f"this moment against the original tab, it may not really be "
                f"simultaneous."
            )
            reachable = [list(best)]

        return reachable

    def _finger_cost(self, to_assignment: list[HandPosition]) -> float:
        """
        Flat preference over finger IDENTITY, independent of stretch or
        shift. This is not a physical-distance measurement (unlike
        stretch/shift) -- it's what stops finger choice from being a
        free variable.

        Confirmed empirically before this existed: because
        `anchor = fret - (finger - 1)`, the DP could reassign a lone
        note's finger purely to relocate its anchor closer to a
        neighboring moment's anchor, making shift_cost artificially
        cheap without the finger choice reflecting anything about how
        a real hand would actually play it (e.g. a single note at fret
        17 assigned to the pinky, anchor 14, just to avoid moving the
        hand -- confirmed via test_behavior.py before this fix, gone
        after it).

        Two components:
          - baseline: `finger_weight` times the PHYSICAL distance from
            the moment's anchor to this note's actual fret (0 for
            index, since anchor is literally defined as index's own
            fret -- see `finger_weight`'s docstring in `__init__` for
            why this replaced an earlier flat per-finger-step version
            that was a real bug, not just a rougher approximation).
          - technique: an EXTRA charge, scaled by `technique_weight`,
            for bend-family/vibrato/slide notes, using per-technique
            magnitude-based cost tables (BEND_FINGER_COST,
            VIBRATO_FINGER_COST, SLIDE_FINGER_COST at module level --
            see their docstrings there for why bending gets a real
            gradient while vibrato/slide are close to finger-agnostic
            except for pinky). This is a real preference order, not
            just "avoid the pinky": bending is a STRENGTH/control
            problem (which finger can push a string in tune, braced by
            the fingers behind it), not the reach problem the baseline
            above prices, so it can cost index (which the baseline
            treats as free) MORE than ring for a bend even though index
            reaches its own fret at zero physical cost. Priced on top
            of, not instead of, the baseline. Left as flat constants on
            purpose, same as before: unlike the baseline, these aren't
            standing in for a physical reach, they're technique-
            difficulty preferences (a ring-finger bend isn't
            meaningfully easier at fret 5 than at fret 15), so there's
            no physical-distance law they should be shrinking to match.
            The bend check and the vibrato check are mutually exclusive
            (`elif`) since both live on the same `Note.modifier` field
            and a note can only carry one modifier. The slide check is
            a separate, independent `if`: sliding lives on
            `Note.arrival` (how you got to this note), not `modifier`
            (how it's decorated once you're on it) -- a note's arrival
            and modifier are independent fields and can both apply to
            the same note (e.g. slide into a bend), so it isn't an
            `elif` off the other two.
        """
        cost = 0.0
        for hp in to_assignment:
            if hp.finger is None or hp.anchor_fret is None:
                continue
            cost += self.finger_weight * physical_distance(hp.anchor_fret, hp.note.fret)
            if hp.note.modifier in BEND_FAMILY:
                cost += self.technique_weight * BEND_FINGER_COST[hp.finger]
            elif hp.note.modifier == Technique.VIBRATO:
                cost += self.technique_weight * VIBRATO_FINGER_COST[hp.finger]
            if hp.note.arrival in SLIDE_TECHNIQUES:
                cost += self.technique_weight * SLIDE_FINGER_COST[hp.finger]
        return cost

    def _slide_continuity_cost(
        self,
        from_assignment: list[HandPosition],
        to_assignment: list[HandPosition],
    ) -> float:
        """
        A slide is one continuous motion of a single finger sliding
        along the string, not a reposition -- you cannot switch which
        finger is pressing down partway through. If `to_assignment`
        puts a DIFFERENT finger on a slide note than whichever finger
        was on that same string a moment ago, that's not a real slide
        at all, so it's charged `slide_continuity_weight`. This is
        distinct from `_finger_cost`'s SLIDE_FINGER_COST, which prices
        WHICH finger is doing the sliding (pinky worse); this prices
        whether the finger is even the SAME one as before, which
        `_finger_cost` has no way to know (it only ever looks at
        `to_assignment` in isolation, never the previous moment).

        Priced high, same as `_legato_conflict_cost` below: this is
        close to a hard physical constraint, not a soft comfort
        preference, so it should only ever lose to an option that's
        even worse (e.g. the origin note genuinely isn't reachable by
        any single finger available, a data problem upstream, not
        something this cost is meant to paper over).
        """
        if not from_assignment:
            return 0.0
        from_finger_by_string = {
            hp.note.string_name: hp.finger for hp in from_assignment if hp.finger is not None
        }
        cost = 0.0
        for hp in to_assignment:
            if hp.finger is None or hp.note.arrival not in SLIDE_TECHNIQUES:
                continue
            prior_finger = from_finger_by_string.get(hp.note.string_name)
            if prior_finger is not None and prior_finger != hp.finger:
                cost += self.slide_continuity_weight
        return cost

    def _legato_conflict_cost(
        self,
        from_assignment: list[HandPosition],
        to_assignment: list[HandPosition],
    ) -> float:
        """
        The opposite requirement from sliding: a hammer-on or pull-off
        needs a SECOND finger, distinct from whatever finger is already
        down on that string, because that first finger is what's
        sounding (or was just sounding) the origin note. If
        `to_assignment` reuses the SAME finger that was on that string
        a moment ago for a hammer-on/pull-off note, that's not
        physically playable as a legato move, so it's charged
        `legato_conflict_weight`.
        """
        if not from_assignment:
            return 0.0
        from_finger_by_string = {
            hp.note.string_name: hp.finger for hp in from_assignment if hp.finger is not None
        }
        cost = 0.0
        for hp in to_assignment:
            if hp.finger is None or hp.note.arrival not in HAMMER_PULL_TECHNIQUES:
                continue
            prior_finger = from_finger_by_string.get(hp.note.string_name)
            if prior_finger is not None and prior_finger == hp.finger:
                cost += self.legato_conflict_weight
        return cost

    def _pulloff_prepositioning_cost(
        self,
        from_assignment: list[HandPosition],
        to_assignment: list[HandPosition],
    ) -> float:
        """
        A stricter, SEPARATE check that applies only to PULL_OFF, not
        HAMMER_ON. A pull-off requires the destination finger to
        already be resting on the string BEFORE the origin note
        releases -- both fingers coexist on the fretboard for an
        instant, like a phantom two-note chord, even though only one
        note sounds at a time. That means the origin and destination
        frets must be within one hand's real physical reach of each
        other -- the same stretch-feasibility test
        `candidate_assignments` already runs for a genuine simultaneous
        dyad, just applied here across a moment boundary instead of
        within one moment. Hammer-on has no equivalent requirement:
        the hammering finger can arrive fresh, it doesn't need to
        already be in position.

        Scaled by how far PAST one hand's reach the two frets are, not
        a flat penalty -- a marginal overreach costs less than a
        genuinely impossible one (e.g. a written "pull-off" spanning
        12 frets, which likely indicates a moments.py grouping issue
        or a parser misread rather than a real playable pull-off, and
        should be priced as unmistakably bad, not just mildly
        discouraged).
        """
        if not from_assignment:
            return 0.0
        max_physical_stretch = physical_distance(0, self.max_stretch_frets - 1)
        from_by_string = {
            hp.note.string_name: hp for hp in from_assignment if hp.anchor_fret is not None
        }
        cost = 0.0
        for hp in to_assignment:
            if hp.anchor_fret is None or hp.note.arrival != Technique.PULL_OFF:
                continue
            prior_hp = from_by_string.get(hp.note.string_name)
            if prior_hp is None:
                continue
            span = physical_distance(prior_hp.anchor_fret, hp.anchor_fret)
            overage = max(0.0, span - max_physical_stretch)
            cost += self.pulloff_prepositioning_weight * overage
        return cost

    def transition_cost(
        self,
        from_assignment: list[HandPosition],
        to_assignment: list[HandPosition],
        rest_time: float,
    ) -> float:
        """
        Cost to move from `from_assignment`'s hand shape to
        `to_assignment`'s, given `rest_time` seconds available to make
        the move (the inter-onset interval -- see module docstring for
        why that's the right quantity, not literal silence).

        `from_assignment == []` means "cold start" (no prior hand
        position, e.g. the very first moment of the song) -- shift cost
        is skipped entirely in that case, only the new shape's own
        stretch and finger cost apply. The same is true of
        `_slide_continuity_cost`, `_legato_conflict_cost`, and
        `_pulloff_prepositioning_cost` below -- all three compare
        `to_assignment` against the PREVIOUS moment's finger-per-string
        layout, so all three are no-ops on a cold start, same as shift
        cost, for the same reason (nothing to compare against yet).

        These three costs are about which finger is PHYSICALLY
        POSSIBLE given what finger was on the same string a moment
        ago -- distinct from `_finger_cost`'s per-note preferences
        (finger_weight/technique_weight), which only ever look at
        `to_assignment` in isolation and have no way to see the
        previous moment at all.
        """
        to_anchors = [hp.anchor_fret for hp in to_assignment if hp.anchor_fret is not None]

        # Stretch cost: how uncomfortable is `to_assignment`'s shape, by
        # itself. Meaningless (and skipped) for a single-note moment.
        if len(to_anchors) >= 2:
            stretch_cost = self.stretch_weight * physical_distance(max(to_anchors), min(to_anchors))
        else:
            stretch_cost = 0.0

        # Finger cost: which finger got used, independent of spatial
        # distance -- see _finger_cost docstring for why this exists.
        finger_cost = self._finger_cost(to_assignment)

        # Shift cost: how far the hand's anchor had to travel to get
        # here, discounted the more rest time was available.
        from_anchors = [hp.anchor_fret for hp in from_assignment if hp.anchor_fret is not None]
        if not from_anchors or not to_anchors:
            # Cold start, or an open-strings-only moment on one side --
            # nothing to measure a physical shift against.
            shift_cost = 0.0
        else:
            from_pos = sum(from_anchors) / len(from_anchors)
            to_pos = sum(to_anchors) / len(to_anchors)
            shift_distance = physical_distance(from_pos, to_pos)
            # Hyperbolic discount: more rest time -> cheaper repositioning,
            # never negative, never fully free (a truly instant jump
            # between rapid-fire notes stays at full cost). Floored at
            # `min_shift_discount`: rhythm.py's rest_time is a heuristic,
            # not a real transcription, so an overestimated rest_time
            # could otherwise push the discount toward zero and make a
            # hand relocation look nearly free purely because the timing
            # guess was generous. The floor is a hedge against bad
            # timing data, not a claim about real minimum movement cost.
            discount = max(self.min_shift_discount, 1.0 / (1.0 + max(rest_time, 0.0)))
            shift_cost = self.shift_weight * shift_distance * discount

        return (
            stretch_cost
            + shift_cost
            + finger_cost
            + self._slide_continuity_cost(from_assignment, to_assignment)
            + self._legato_conflict_cost(from_assignment, to_assignment)
            + self._pulloff_prepositioning_cost(from_assignment, to_assignment)
        )

    def solve(self, moments: list[Moment]) -> list[list[HandPosition]]:
        """
        Forward-compute / backward-trace DP over the whole song.
        dp_cost[i][k] = cheapest total cost to have played moments
        0..i, ending in candidate_assignments(moments[i])[k]. Each node
        remembers which predecessor achieved that cheapest cost
        (dp_back), so after the forward pass reaches the last moment,
        following those pointers backward from the globally cheapest
        final node recovers the whole song's fingering in one pass --
        an early note can "choose" a finger based on something several
        notes later without ever explicitly looking ahead itself,
        because the DP already explored every path before committing.
        """
        if not moments:
            return []

        for m in moments:
            if m.start_time is None or m.duration is None:
                raise ValueError(
                    f"Moment at column {m.column} has no start_time/duration -- "
                    f"run rhythm.assign_timing() before optimizer.solve()."
                )

        layer_candidates = [self.candidate_assignments(m) for m in moments]
        for i, cands in enumerate(layer_candidates):
            if not cands:
                # Every path through candidate_assignments (normal,
                # too-wide-a-stretch fallback, too-many-fingers
                # fallback) returns at least one candidate -- this
                # should be unreachable. Fail loudly with the moment
                # that broke the invariant rather than a cryptic
                # IndexError several stack frames later in the
                # backward trace.
                raise RuntimeError(
                    f"candidate_assignments returned zero options for the "
                    f"moment at column {moments[i].column} -- this is a bug "
                    f"in candidate_assignments, not a data problem."
                )

        dp_cost: list[list[float]] = [
            [self.transition_cost([], cand, rest_time=0.0) for cand in layer_candidates[0]]
        ]
        dp_back: list[list[int | None]] = [[None] * len(layer_candidates[0])]

        for i in range(1, len(moments)):
            rest_time = moments[i - 1].duration  # inter-onset interval, see module docstring
            costs_i: list[float] = []
            backs_i: list[int | None] = []

            for cand in layer_candidates[i]:
                best_cost = math.inf
                best_prev = None
                for prev_idx, prev_cand in enumerate(layer_candidates[i - 1]):
                    cost = dp_cost[i - 1][prev_idx] + self.transition_cost(prev_cand, cand, rest_time)
                    if cost < best_cost:
                        best_cost = cost
                        best_prev = prev_idx
                costs_i.append(best_cost)
                backs_i.append(best_prev)

            dp_cost.append(costs_i)
            dp_back.append(backs_i)

        best_final = min(range(len(dp_cost[-1])), key=lambda idx: dp_cost[-1][idx])

        path_indices = [0] * len(moments)
        path_indices[-1] = best_final
        for i in range(len(moments) - 1, 0, -1):
            path_indices[i - 1] = dp_back[i][path_indices[i]]

        return [layer_candidates[i][path_indices[i]] for i in range(len(moments))]


if __name__ == "__main__":
    # Small hand-checkable sample, mirroring the real G-fret-7/D-fret-9
    # dyad this project found in Hotel California: a single note, then
    # a dyad, then the same single note again -- the cheap answer is
    # obviously "come back to roughly the same hand position," which is
    # exactly what a correct DP should recover without being told to.
    from src.rhythm import assign_timing

    sample_moments = [
        Moment(notes=[Note(string_index=0, string_name="e", fret=5, column=0)], column=0),
        Moment(
            notes=[
                Note(string_index=2, string_name="G", fret=7, column=20),
                Note(string_index=3, string_name="D", fret=9, column=20),
            ],
            column=20,
        ),
        Moment(notes=[Note(string_index=0, string_name="e", fret=5, column=40)], column=40),
    ]
    assign_timing(sample_moments)

    optimizer = FingeringOptimizer()
    solution = optimizer.solve(sample_moments)

    for moment, assignment in zip(sample_moments, solution):
        desc = ", ".join(
            f"{hp.note.string_name}:fret{hp.note.fret}->finger{hp.finger}(anchor{hp.anchor_fret})"
            for hp in assignment
        )
        print(f"Moment(col={moment.column}, t={moment.start_time:.2f}s): {desc}")