"""
audio.py

Synthesizes audio for the optimizer's chosen fingering, so the
visualizer isn't silent -- the stated goal is for someone to actually
LEARN from this, and hearing the note land with the picture makes that
real in a way a silent dot never will.

METHOD: additive synthesis with a swept fundamental
-------------------------------------------------------
A real plucked string is closer to Karplus-Strong (a decaying noise
loop), but Karplus-Strong's pitch is fixed by its delay-line length --
changing pitch mid-note means changing the delay line mid-flight, which
introduces audible clicks/artifacts. Since bends are the entire point
here, this instead builds each note as a small stack of harmonics
(fundamental + 2nd + 3rd, guitar-ish weighting) whose *frequency can be
swept smoothly sample-by-sample* by integrating an instantaneous
frequency curve into phase. That's what makes a clean, click-free bend
possible. It's a plainer, more "synth-guitar" timbre than a sampled
string -- an honest, explainable tradeoff for glide-ability, not an
attempt to sound identical to a real guitar recording.

The same swept-frequency machinery gives slides a real glide too: a
SLIDE_UP/SLIDE_DOWN (or unresolved bare SLIDE) note's frequency curve
ramps from the DEPARTING note's pitch (see
`src.moments.previous_note_by_string` -- the same lookup `visualizer.py`
uses to animate the dot) to this note's own pitch over the first half
of its hold, instead of jumping straight to the landed pitch the way a
plain note does.
"""

import math
import os

import numpy as np

from src.moments import previous_note_by_string
from src.parser import MAX_REALISTIC_FRET, Technique

SAMPLE_RATE = 44100

# Open-string frequencies (Hz), standard tuning, indexed the same way
# parser.py's STANDARD_TUNING_ORDER is: 0=e (high), 5=E (low).
OPEN_STRING_HZ = {
    0: 329.63,  # e4
    1: 246.94,  # B3
    2: 196.00,  # G3
    3: 146.83,  # D3
    4: 110.00,  # A2
    5: 82.41,   # E2
}

BEND_TECHNIQUES = {Technique.BEND, Technique.REBEND, Technique.RELEASE}
VIBRATO_TECHNIQUES = {Technique.VIBRATO}
# Mirrors visualizer.py's SLIDE_TECHNIQUES -- bare Technique.SLIDE is
# included because parser.py can only resolve its up/down direction by
# comparing against a previous note on the same LINE, so a slide
# opening a new block/line can still reach here unresolved; the actual
# glide math below only needs a start fret, not a direction label, so
# it doesn't care which of the three this is.
SLIDE_TECHNIQUES = {Technique.SLIDE_UP, Technique.SLIDE_DOWN, Technique.SLIDE}

# How many frets an unanchored slide (no previous note on this string
# to read a real starting pitch from) runs before landing -- mirrors
# visualizer.py's UNANCHORED_SLIDE_RUN_FRETS so the ear and the eye
# agree even in this fallback case.
UNANCHORED_SLIDE_RUN_FRETS = 8


def fret_to_hz(string_index: int, fret: float) -> float:
    """Equal temperament: each fret is one semitone, so N frets up is
    open_hz * 2**(N/12). Works for fractional "frets" too, which is
    how a bend's in-between pitch is expressed below."""
    open_hz = OPEN_STRING_HZ.get(string_index, OPEN_STRING_HZ[5])
    return open_hz * (2.0 ** (fret / 12.0))


def _bend_target_fret(note) -> float:
    """Mirrors visualizer.py's `_bend_target_fret` -- same reading of
    an unqualified bend as a full/whole-step (2 frets) unless the tab
    says otherwise. Kept as a separate copy (not imported) because
    audio.py has no other dependency on visualizer.py and shouldn't
    need one just for this."""
    if note.technique_target is not None:
        return note.technique_target
    if note.bend_amount is not None:
        return note.fret + note.bend_amount * 2
    return note.fret + 2


def _frequency_curve(note, n_samples: int, slide_start_fret: float | None = None) -> np.ndarray:
    """Per-sample instantaneous frequency (Hz) for one note's duration,
    honoring bends/vibrato/slides as an actual pitch sweep -- not just a
    visual cue this time.

    `slide_start_fret` (None unless this note arrived via a slide) is
    handled as an ADDITIVE offset on top of whatever the modifier-based
    curve below already computed, decaying to 0 over the first half of
    the hold -- see `_note_offset_at`'s docstring in visualizer.py for
    the matching dx animation. Additive, not an alternate branch, so a
    slide INTO a bent/vibrato'd note (rare, but the tab format allows
    it) still gets both effects instead of one silently overriding the
    other."""
    progress = np.linspace(0.0, 1.0, n_samples, endpoint=False)

    if note.modifier == Technique.PRE_BEND:
        fret_curve = np.full(n_samples, _bend_target_fret(note))
    elif note.modifier in BEND_TECHNIQUES:
        target = _bend_target_fret(note)
        ramp = np.clip(progress / 0.4, 0.0, 1.0)
        if note.modifier == Technique.RELEASE:
            ramp = 1.0 - ramp
        fret_curve = note.fret + (target - note.fret) * ramp
    elif note.modifier in VIBRATO_TECHNIQUES:
        # +/- a quarter-tone-ish wobble, ~6 Hz -- same rate the
        # visualizer's dot jitters at, so the ear and the eye agree.
        fret_curve = note.fret + 0.15 * np.sin(2 * math.pi * 6.0 * progress)
    else:
        fret_curve = np.full(n_samples, float(note.fret if note.fret >= 0 else 0))

    if slide_start_fret is not None:
        # Ramp over the first half of the hold -- a real slide is a
        # quick, continuous slap up/down the neck, not a slow squeeze
        # like a bend -- then hold at 0 offset (i.e. the note's own
        # landed pitch) for the rest.
        ramp = np.clip(progress / 0.5, 0.0, 1.0)
        fret_curve = fret_curve + (slide_start_fret - note.fret) * (1.0 - ramp)

    return fret_to_hz(note.string_index, fret_curve)


def synth_note(
    note, duration: float, sample_rate: int = SAMPLE_RATE, slide_start_fret: float | None = None
) -> np.ndarray:
    """Renders one note (any technique) to a mono float32 buffer,
    `duration` seconds long. Muted/dead notes (fret < 0) get a short
    burst of filtered noise (a pick "chuck") instead of a pitched
    tone -- there's no pitch to synthesize for those.

    `slide_start_fret`: see `_frequency_curve` -- pass this whenever
    `note.arrival` is a slide technique; `render_track` resolves it via
    `src.moments.previous_note_by_string` before calling this."""
    n_samples = max(1, int(duration * sample_rate))
    t = np.arange(n_samples) / sample_rate
    envelope = np.exp(-3.5 * t)  # plucked-string-ish decay, fast attack implied by starting at 1.0

    # A short, decay-independent fade over the LAST few ms of the
    # buffer. Without this, a short note (dense passages -- a 16th
    # note at tempo can be ~0.05s) gets truncated mid-waveform when its
    # slot ends, since exp(-3.5t) alone hasn't decayed to ~0 yet at
    # that point. That sample-to-sample jump to the next note's
    # buffer is an audible click; stacked across a whole dense passage,
    # many overlapping clicks read as a "muffled crashing" noise
    # instead of distinct notes -- this is what was actually happening,
    # not a bug in which notes get synthesized.
    fade_samples = min(n_samples, max(1, int(0.008 * sample_rate)))
    fade = np.ones(n_samples)
    fade[n_samples - fade_samples:] = np.linspace(1.0, 0.0, fade_samples)
    envelope = envelope * fade

    if note.fret is None or note.fret < 0:
        noise = np.random.default_rng(hash((note.string_index, note.column)) & 0xFFFF).standard_normal(n_samples)
        return (noise * envelope * 0.25).astype(np.float32)

    freq_curve = _frequency_curve(note, n_samples, slide_start_fret)
    phase = 2 * math.pi * np.cumsum(freq_curve) / sample_rate
    wave = 0.60 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
    return (wave * envelope).astype(np.float32)


def render_track(moments, assignments, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Mixes every note in every moment into one full-song mono track,
    each placed at its real `start_time`/`duration` from rhythm.py --
    chords/dyads sum naturally since they share a start_time. Output is
    peak-normalized so louder passages don't clip."""
    if not moments:
        return np.zeros(1, dtype=np.float32)

    total_duration = moments[-1].start_time + moments[-1].duration
    total_samples = max(1, int(total_duration * sample_rate) + sample_rate)  # +1s tail for the last note's decay
    track = np.zeros(total_samples, dtype=np.float32)

    # Same cross-moment "what note came before this one on this
    # string" lookup visualizer.py uses to animate a slide's dot --
    # here it supplies the pitch a slide glides FROM.
    prev_note_map = previous_note_by_string(moments)

    for moment, assignment in zip(moments, assignments):
        start_sample = int(moment.start_time * sample_rate)
        for hp in assignment:
            note = hp.note
            slide_start_fret = None
            if note.arrival in SLIDE_TECHNIQUES:
                prev_note = prev_note_map.get(id(note))
                if prev_note is not None and prev_note.fret >= 0:
                    slide_start_fret = prev_note.fret
                else:
                    # No real previous note on this string -- run a
                    # fixed distance in whatever direction is known (or
                    # up, as a last resort), clamped to a realistic
                    # fret range. Mirrors visualizer.py's fallback so
                    # the ear and the eye still agree even here.
                    direction = -1 if note.arrival == Technique.SLIDE_DOWN else 1
                    fallback = note.fret - direction * UNANCHORED_SLIDE_RUN_FRETS
                    slide_start_fret = max(0, min(MAX_REALISTIC_FRET, fallback))
            note_audio = synth_note(note, moment.duration, sample_rate, slide_start_fret)
            end_sample = min(total_samples, start_sample + len(note_audio))
            track[start_sample:end_sample] += note_audio[: end_sample - start_sample]

    peak = np.max(np.abs(track))
    if peak > 1e-6:
        track = track / peak * 0.9
    return track


def _pcm16_bytes(track: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Builds an in-memory 16-bit PCM mono WAV, as bytes."""
    import io
    import wave

    pcm = np.clip(track, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm16.tobytes())
    return buf.getvalue()


def write_wav(path: str, track: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Writes a 16-bit PCM mono WAV to disk -- stdlib `wave`, no extra
    dependency (numpy is already required by visualizer.py/optimizer.py)."""
    with open(path, "wb") as f:
        f.write(_pcm16_bytes(track, sample_rate))


def write_temp_wav(track: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """
    Writes `track` to a throwaway temp WAV file and returns its path.
    Used by the interactive viewer to play an arbitrary SLICE of the
    full song's audio (e.g. "from t=42s onward") -- `winsound` can only
    play a wav file from its own beginning, it has no seek/offset
    argument, so resuming playback from the middle of a song means
    handing it a fresh little file that already starts at the right
    place, not the whole-song file. Not cleaned up automatically (best
    left on disk in the OS temp dir rather than risk deleting a file
    winsound might still be reading asynchronously); harmless clutter,
    the OS temp dir gets swept eventually.
    """
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".wav", prefix="fretboard_audio_")
    os.close(fd)
    write_wav(path, track, sample_rate)
    return path


# How much earlier the animation's clock should be started relative to
# the moment playback is triggered, to compensate for real-world
# playback startup latency -- see `play_wav_async`'s docstring for
# where this number comes from and how to retune it.
PLAYBACK_LATENCY_SECONDS = 0.09


def play_wav_async(path: str) -> None:
    """
    Fire-and-forget playback, Windows-only (this project's dev
    environment) via the stdlib `winsound` module -- no extra
    dependency. Silently no-ops on other platforms rather than
    crashing the visualizer over a nice-to-have.

    WHY THE DOTS AND AUDIO CAN DRIFT A LITTLE
    ------------------------------------------
    `winsound.PlaySound(path, ...SND_FILENAME...)` re-opens and reads
    the wav FILE from disk every time it's called, and the OS audio
    device needs a moment to actually start producing sound after
    that -- both happen after "Play was clicked" but before sound is
    actually audible, while the animation's clock is already running.
    `PLAYBACK_LATENCY_SECONDS` delays starting the timer by that much
    to absorb the gap -- tune it up/down by ear if it's still off on a
    given machine, there's no portable way to measure it exactly from
    Python without native audio APIs.

    A `SND_MEMORY | SND_ASYNC` version (skip the disk read by playing
    already-loaded bytes) was tried and reverted: `winsound` on this
    Python build raises `RuntimeError: Cannot play asynchronously from
    memory` for that combination -- asynchronous playback from an
    in-memory buffer isn't actually supported here, so `SND_FILENAME`
    is the one that reliably works, not a downgrade for its own sake.
    """
    try:
        import winsound

        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except ImportError:
        pass


def stop_playback() -> None:
    try:
        import winsound

        winsound.PlaySound(None, winsound.SND_PURGE)
    except ImportError:
        pass


if __name__ == "__main__":
    from src.parser import parse_tab
    from src.moments import group_into_moments
    from src.rhythm import assign_timing
    from src.optimizer import FingeringOptimizer

    with open("data/hotel_california_solo.txt", encoding="utf-8") as f:
        raw = f.read()
    lines = raw.split("\n")
    section = "\n".join(lines[5:11])

    notes = parse_tab(section)
    moments = assign_timing(group_into_moments(notes), bpm=75)
    solution = FingeringOptimizer().solve(moments)

    track = render_track(moments, solution)
    write_wav("hotel_california_demo.wav", track)
    print(f"Saved hotel_california_demo.wav ({len(track) / SAMPLE_RATE:.1f}s)")
