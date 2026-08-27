"""
optimizer.py

Chooses one hand-fingering assignment per Moment for the whole song,
minimizing total physical effort.

PHYSICAL MODEL
--------------
The hand sits at an ANCHOR FRET; each finger naturally covers one
fret from there (index=anchor, middle=+1, ring=+2, pinky=+3), so
anchor = fret - (finger - 1). A dyad is a JOINT decision -- fingers
must differ (barre exception: same finger, same fret) and be
reachable from one anchor at once. That's why the DP's nodes are
whole-moment assignments, not per-note choices.

THREE COSTS -- all in PHYSICAL fret distance, not raw fret-count
(frets narrow toward the body, so a 2-fret jump at fret 3 is a real
reach; the same jump at fret 15 is nearly free -- everything goes
through `physical_fret_position`, never raw subtraction):
- STRETCH: spread within one moment's own shape (stretch_weight).
- SHIFT: how far the anchor moves between moments, discounted by
  available rest_time (shift_weight).
- FINGER: which finger got used, independent of stretch/shift.
  Needed because `anchor = fret - (finger-1)` makes finger choice a
  free variable otherwise -- the DP would use a far finger just to
  fake a cheaper anchor for shift_cost. Originally priced as a FLAT
  per-finger constant -- that was a real bug: a flat cost never
  shrinks, but shift's physical distance does, so there was always
  some fret position where relocating beat switching fingers purely
  from a units mismatch. Pricing finger cost as a physical reach too
  fixes that -- all three costs now shrink together, so the tradeoff
  is decided by real rest_time, not raw position. technique_weight is
  the one deliberate flat exception: it's a technique-difficulty
  preference (e.g. pinky on a bend), not a spatial one.

REST_TIME = the inter-onset interval (previous moment's duration),
not literal silence -- how much time is available to move before the
next note must sound.

THE DP
------
Layer = moment. Node = one joint HandPosition assignment
(candidate_assignments). Edge cost = transition_cost. solve() runs
the standard forward pass (cheapest cost to reach each node, plus a
back-pointer) then backtraces from the cheapest final node -- same
shape as edit-distance / Viterbi, "hand shape" instead of "hidden
state."

"""

import itertools
import math
from dataclasses import dataclass

from src.moments import Moment
from src.parser import Note, Technique

# parser.py's parse_line only ever attaches
# ONE modifier token to a Note (whichever immediately follows its fret
# token) -- so a real "bend then release" ("11b13r11") ends up with
# modifier=BEND only; the RELEASE token is silently orphaned and
# dropped before it ever reaches a Note. This is the limitation of this 
# project's parser that's not really worth because of non-standardized tab notation, 
# and the optimizer is written to handle it anyway: if a Note ever did carry both a BEND and a RELEASE
# the optimizer would still treat it as a bend-family technique (the RELEASE is just a decoration on
# top of the bend, not a separate physical motion).
BEND_FAMILY = (Technique.BEND, Technique.PRE_BEND, Technique.REBEND, Technique.RELEASE)


#     BEND_FINGER_COST: bending is a sustained strength/control
#     problem (holding a string under tension in tune). 
#     ring finger is usually easiest (the two fingers behind it
#     on the neck can brace it), then middle, then index (weakest of
#     the three without help behind it), with pinky a much BIGGER jump
#     than the even spacing between the others (weakest finger
#     overall, unless you are like that one guy who can bend with his pinky, 
#     in which case you are a freak and this project is not for you).
#     VIBRATO_FINGER_COST: vibrato is an bascially good for all fingers
#     except pinky. 
# Both deliberately do NOT match the general `_finger_cost` baseline's
# index-is-always-cheapest assumption -- that baseline is about
# physical REACH from the anchor (index reaches furthest for free
# because anchor is defined by it), which has nothing to do with which
# finger has the STRENGTH/control to bend or oscillate a string well.
# A lone bent note with plenty of rest_time around it should still
# prefer ring over index even though index "reaches for free"

BEND_FINGER_COST = {3: 0.0, 2: 0.3, 1: 0.6, 4: 1.2}  # ring, middle, index, pinky
VIBRATO_FINGER_COST = {3: 0.0, 2: 0.0, 1: 0.0, 4: 1.0}  # only pinky is penalized

# Sliding lives on Note.arrival (how you got to this note). Bare
# Technique.SLIDE is included alongside SLIDE_UP/SLIDE_DOWN because
# parser.py's parse_line only resolves "s" into a direction when
# there's a prior note on the same string to compare against -- the
# first note on a string with a leading "s" has no prior note yet and
# can reach here still unresolved. Like vibrato, sliding good for all 
# fingers except pinky.
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
    # Physical distance between two frets, in the same scale-length units.
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
                            moment's own hand shape (chord difficulty). #does this have vertical distance?
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
      technique_weight   -- multiplier applied to the per-technique,
                            per-finger MAGNITUDE cost tables
                            (BEND_FINGER_COST, VIBRATO_FINGER_COST,
                            SLIDE_FINGER_COST, module level) for
                            bending, vibrato, and sliding respectively
                            -- see those tables' own docstrings for why
                            bending gets a real gradient (ring best,
                            pinky a much bigger jump than the rest)
                            while vibrato/slide is only for pinky.
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
                            this is a limitation because there is no strict bpm
                            and rhythm recording in textual tabs, thus good conclusion
                            cannot be drawn from this project about the actual time it takes to move the hand.
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
        shift_weight: float = 2.5,  # too much awkward repositioning thats why i increased it from 1.0 to 2.5
        max_stretch_frets: float = 4.5,
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

        Candidate counts grow at high frets (more real options genuinely exist), 
        which is expected and still entirely tractable at solo-tab length.

        Index (finger 1) is excluded from this range logic on purpose:
        `anchor_fret` is DEFINED as wherever the index finger sits
        (see the module docstring's "THE PHYSICAL MODEL" section).
        """
        if note.fret <= 0:
            return [HandPosition(note=note, finger=None, anchor_fret=None)]

        options = [HandPosition(note=note, finger=1, anchor_fret=note.fret)]
        for finger in FINGERS[1:]:
            # anchor is depended on the index position, thats why we are excluding index finger
            # from the loop. in real playing, especially in higher frets, the index can sometimes
            # reach a note that's 2-3 frets away, but in our model we are simplyifing it 
            max_reach = physical_distance(0, finger - 1) 
            # max reach acts as a ceiling measured from the nut to the finger
            offset = finger - 1  # traditional minimum, always kept as the floor
            while True:
                anchor = note.fret - offset
                if anchor < 0:
                    break  # can't reach this fret from any real hand position
                if physical_distance(anchor, note.fret) > max_reach:
                    break  # past this finger's real physical reach limit at this position
                options.append(HandPosition(note=note, finger=finger, anchor_fret=anchor))
                offset += 1 # keep trying higher anchors until we hit the physical limit
        return options

    # the returned inner list is a single joint assignment for the whole moment
    # and it's a list because there can be multiple notes in one moment
    # the returned outer list collects different combos of joint assignments for the same moment
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

        A moment with more DISTINCT fretted positions than 4 
        can't satisfy rule 1 at all, by pigeonhole: there just aren't
        enough fingers, independent of stretch. That's genuinely out
        of this project's scope and, on a solo tab, almost certainly a
        moments.py grouping artifact rather than a real chord. Handled
        the same way as the stretch fallback below: don't crash the
        whole song, use a placeholder, warn loudly.

        NOTE ON RAW NOTE COUNT vs. DISTINCT FRET COUNT 
        an earlier version of this check rejected any
        moment with more than 4 fretted NOTES, full stop -- that's
        wrong for a real full BARRE CHORD. Rejecting on raw note count
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

        
        max_physical_stretch = physical_distance(0, self.max_stretch_frets - 1)

        # plan b if no reachable combination exists
        finger_legal: list[tuple[HandPosition, ...]] = []
        # plan a if a reachable combination exists
        reachable: list[list[HandPosition]] = []

        # * unpacks the list of lists into separate arguments for product()
        for combo in itertools.product(*per_note_options):
            # A finger can be reused across notes in the same moment
            # ONLY as a barre: the same finger pressing the
            # SAME fret on multiple strings
            # Reusing a finger across two DIFFERENT frets is impossible 
            # and is rejected
            finger_frets: dict[int, int] = {}
            barre_conflict = False
            for hp in combo:
                if hp.finger is None:
                    continue
                # check if finger is already used earlier in same combo
                if hp.finger in finger_frets and finger_frets[hp.finger] != hp.note.fret:
                    barre_conflict = True
                    break
                finger_frets[hp.finger] = hp.note.fret
            # if we found a barre conflict, skip this combo and continue to the next one
            if barre_conflict:
                continue

            finger_legal.append(combo)

            anchors = [hp.anchor_fret for hp in combo if hp.anchor_fret is not None]
            # if there are 2+ anchors, calculate physical distance btw highest and lowest
            # if there's only 1 anchor, spread is 0.0
            spread = physical_distance(max(anchors), min(anchors)) if len(anchors) >= 2 else 0.0
            if spread <= max_physical_stretch:
                reachable.append(list(combo))

        if not finger_legal:
            # If literally no combo survived the finger-conflict check at all
            
            # count how many notes in this moment are fretted (fret > 0)
            fretted_count = sum(1 for n in moment.notes if n.fret > 0)
            # a fake placeholder assignment that uses index for every fretted note, and None for open strings
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

        # If some combos were finger-legal but none fit within stretch range
        if not reachable:
            # Pick whichever finger-legal combo has the smallest stretch
            best = min(
                finger_legal,
                key=lambda combo: (
                    max((hp.anchor_fret for hp in combo if hp.anchor_fret is not None), default=0)
                    - min((hp.anchor_fret for hp in combo if hp.anchor_fret is not None), default=0)
                ),
            )
            notes_desc = [(n.string_name, n.fret) for n in moment.notes]
            # readable description of the notes involved, and print a warning explaining that no combo comfortably fits
            print(
                f"WARNING: column {moment.column}: no finger combination for "
                f"this moment fits within one hand span (max_stretch_frets="
                f"{self.max_stretch_frets}). Notes involved: {notes_desc}. "
                f"Falling back to the least-stretched option anyway -- verify "
                f"this moment against the original tab, it may not really be "
                f"simultaneous."
            )
            # Set reachable to just this one best-effort combo, so the function still has something to return
            reachable = [list(best)]

        return reachable

    # to_assignment is a list of HandPosition, one per note in the moment.
    # to_assignment acts as a NODE
    # to_assignment is the candidate of CURRENT moment
    # from_assignment is the candidate of PREVIOUS moment
    def _finger_cost(self, to_assignment: list[HandPosition]) -> float:
        """
    Cost for finger identity, independent of stretch/shift -- stops
    the DP from picking an unrealistic finger just to cheapen
    shift_cost (see finger_weight's docstring in __init__).

    Two parts per note:
      - baseline: finger_weight * physical reach from anchor to fret
        (0 for index, since anchor IS index's fret).
      - technique: extra cost for bend/vibrato/slide, from the
        magnitude tables above. Bend/vibrato share Note.modifier
        (elif -- mutually exclusive); slide is Note.arrival, a
        separate field that can co-occur with either (plain if).
        """
        cost = 0.0
        for hp in to_assignment:
           # skip open strings and muted notes
            # Not zero at a finger's own canonical offset (eg. ring at 2) on purpose:
        # tried charging only for reach BEYOND canonical, and 
        # any nut-calibrated budget hits zero marginal cost at high
        # enough frets, so the hand just parks and stretches forever
        # with progressively higher fingers, for free (a
        # 12->14->15->17 climb never moved at all--although its possible. 
        # Charging the FULL reach, always, is what keeps that loophole closed.

            if hp.finger is None or hp.anchor_fret is None:
                continue
            cost += self.finger_weight * physical_distance(hp.anchor_fret, hp.note.fret)
            # a note can only have 1 modifier
            if hp.note.modifier in BEND_FAMILY:
                cost += self.technique_weight * BEND_FINGER_COST[hp.finger]
            elif hp.note.modifier == Technique.VIBRATO:
                cost += self.technique_weight * VIBRATO_FINGER_COST[hp.finger]
            # a note can have slide on top of other modifiers
            if hp.note.arrival in SLIDE_TECHNIQUES:
                cost += self.technique_weight * SLIDE_FINGER_COST[hp.finger]
        return cost

    def _slide_continuity_cost(
        self,
        from_assignment: list[HandPosition],
        to_assignment: list[HandPosition],
    ) -> float:
        """
    A slide needs the same finger as whatever was on that string a
    moment ago. You can't switch mid-slide. Priced high (near-hard
    constraint), same as _legato_conflict_cost below.
        """
        # if theres no previous note
        if not from_assignment:
            return 0.0
        # this is shortened def of dictionary
        # in form of {string_name: finger}
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
        Opposite of sliding: hammer-on/pull-off needs a different finger
        from whatever was on that string a moment ago.
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
        Why: pull off destination finger must already be on the string
        before the origin note releases, so the two frets must be within
        one hand's reach. 
        this cost scales by overage (how far past the hand's reach the two frets are).
        """
        if not from_assignment:
            return 0.0
        max_physical_stretch = physical_distance(0, self.max_stretch_frets - 1)
        from_by_string = {
            # map to HandPosition because we need anchor_fret too
            hp.note.string_name: hp for hp in from_assignment if hp.anchor_fret is not None
        }
        cost = 0.0
        for hp in to_assignment:
            if hp.anchor_fret is None or hp.note.arrival != Technique.PULL_OFF:
                continue
            prior_hp = from_by_string.get(hp.note.string_name)
            # if it's an open string or silent before
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
        given previous moment and current moment's hand position assignment, 
        how much rest time is available, then return the total cost of transition

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
            # ex: if both index and middle are in fret 5, then their anchor is 5 and maybe 6
            # average anchor is 5.5 then.
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
        # nothing to solve for empty song
        if not moments:
            return []

        for m in moments:
            if m.start_time is None or m.duration is None:
                raise ValueError(
                    f"Moment at column {m.column} has no start_time/duration. "
                    f"run rhythm.assign_timing() before optimizer.solve()."
                )
        # 3 layers of list comprehension: outer = moments, middle = candidate assignments for 
        # that moment, inner = HandPosition for each note in that assignment
        layer_candidates = [self.candidate_assignments(m) for m in moments]
        for i, cands in enumerate(layer_candidates):
            if not cands:
                # Every path through candidate_assignments (normal,
                # too-wide-a-stretch fallback, too-many-fingers
                # fallback) returns at least one candidate -- this
                # should be unreachable. Fail with the moment
                # that broke the invariant rather than a cryptic
                # IndexError several stack frames later in the
                # backward trace.
                raise RuntimeError(
                    f"candidate_assignments returned zero options for the "
                    f"moment at column {moments[i].column} -- this is a bug "
                    f"in candidate_assignments, not a data problem."
                )

        # stores each layer's candidate assignment costs, and the 
        # backtrace pointers to the previous layer's best candidate
        dp_cost: list[list[float]] = [
            [self.transition_cost([], cand, rest_time=0.0) for cand in layer_candidates[0]]
        ]
        # pointer to which candidate in the previous layer was cheapest for each candidate in this layer
        # the value is which candidate number, in the previous moment, was the cheapest
        # thing to come from
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