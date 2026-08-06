"""
visualizer.py

Renders the optimizer's chosen fingering as an animated fretboard:
6 strings, matplotlib, one colored dot per currently-sounding note,
played back at the REAL timing computed by rhythm.py.

SCOPE, ON PURPOSE (matches PLAN.md — no finger/hand animation this
cycle, dots only)
--------------------------------------------------------------------
This does not draw a hand or fingers. Each note is a colored dot at
its (string, fret) position; the color encodes WHICH finger the
optimizer chose. Bends and vibrato move the dot VERTICALLY, off the
string line, toward a neighboring string — matching how a real bend
actually looks (the string deflects sideways relative to its own
length), not by sliding the dot up the neck to a different fret's x
position, which is not what a bend looks like:
  - a BEND/REBEND note's dot deflects toward its neighboring string
    over the first part of its held duration, then stays deflected
  - a PRE_BEND note's dot starts already deflected (the string was
    bent before it was picked)
  - a RELEASE note's dot starts deflected and ramps back down to the
    string's rest line — undoing an existing bend
  - deflection direction is toward the string closer to the high e
    (the standard way a bend is physically pushed), except on the high
    e string itself, which has nothing above it to push into and
    bends toward B instead
  - a VIBRATO note's dot jitters vertically in place for its duration
This is a deliberately simple, honest stand-in for "the pitch is moving
during this note" — not a physically simulated string.

FRET POSITIONS REUSE THE OPTIMIZER'S PHYSICAL MODEL
------------------------------------------------------
`physical_fret_position` (equal-tempered, frets narrowing toward the
body) is the same function optimizer.py uses to judge hand stretch —
reused here so the drawn fretboard is geometrically the same one the
optimizer reasoned about, not a separate made-up layout. A dot for
fret F is drawn at the MIDPOINT between fret wires F-1 and F (where a
real finger actually sits), not on the wire itself.
"""

import bisect
import math
import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from src.moments import Moment
from src.optimizer import HandPosition, physical_fret_position
from src.parser import Technique

NUM_STRINGS = 6
STRING_NAMES_TOP_TO_BOTTOM = ["e", "B", "G", "D", "A", "E"]  # matches parser.py's STANDARD_TUNING_ORDER

# One color per finger, plus open-string/muted. Colorblind-friendly-ish,
# high contrast against a light or dark background.
FINGER_COLORS = {
    1: "#4C9AFF",  # index  -- blue
    2: "#57D9A3",  # middle -- green
    3: "#FFAB00",  # ring   -- amber
    4: "#FF5C5C",  # pinky  -- red
    None: "#999999",  # open string / muted -- gray
}

FINGER_LABELS = {
    1: "1 · index",
    2: "2 · middle",
    3: "3 · ring",
    4: "4 · pinky",
    None: "open string",
}

BEND_TECHNIQUES = {Technique.BEND, Technique.REBEND, Technique.RELEASE}
VIBRATO_TECHNIQUES = {Technique.VIBRATO}


class FretboardVisualizer:
    def __init__(self, num_frets: int = 15):
        self.num_frets = num_frets
        self.fig, self.ax = plt.subplots(figsize=(12, 4))
        self._board_drawn = False
        # Fret/string/legend/time-readout text is sized as a fraction
        # of the figure's actual pixel height rather than a fixed
        # point size, and re-applied on every resize -- so maximizing
        # or dragging the window bigger makes the labels bigger too,
        # instead of the same tiny fixed-size text just floating in
        # more empty space.
        self._fret_texts: list = []
        self._string_texts: list = []
        self._legend = None
        self._time_text = None
        self._anchor_text = None
        self.fig.canvas.mpl_connect("resize_event", self._on_resize)

    def _on_resize(self, _event=None) -> None:
        """Rescales all on-figure text to the window's current size,
        relative to this class's baseline figsize height (4 inches --
        see __init__). Clamped so a tiny window doesn't shrink text to
        unreadable, and a huge one doesn't blow it up absurdly."""
        height_in = self.fig.get_size_inches()[1]
        scale = max(0.6, min(3.0, height_in / 4.0))

        for text, base_size in self._fret_texts:
            text.set_fontsize(base_size * scale)
        for text, base_size in self._string_texts:
            text.set_fontsize(base_size * scale)
        if self._board_drawn:
            # A legend isn't just text -- its handle/marker sizes and
            # spacing are laid out ONCE from the `fontsize` given at
            # creation, so mutating its texts' fontsize afterward
            # (like the plain labels above) leaves the markers and
            # spacing stuck at the original size while only the text
            # grows -- it was rebuilt at the new size instead, which is
            # what actually keeps the whole legend in proportion.
            self._draw_legend(fontsize=8 * scale)
        if self._time_text is not None:
            self._time_text.set_fontsize(9 * scale)
        if self._anchor_text is not None:
            self._anchor_text.set_fontsize(9 * scale)

        self.fig.canvas.draw_idle()

    def _fret_wire_x(self, fret: int) -> float:
        """Physical x-position of fret wire `fret` itself (0 = the nut)."""
        return physical_fret_position(fret)

    def _fret_x(self, fret: int) -> float:
        """
        Where a note at `fret` is actually drawn: the nut for open
        strings, otherwise the MIDPOINT between the fret wire before it
        and the fret wire itself -- matching where a finger really
        presses on a real neck, not the wire line.
        """
        if fret <= 0:
            return 0.0
        return (self._fret_wire_x(fret - 1) + self._fret_wire_x(fret)) / 2

    def _string_y(self, string_index: int) -> float:
        """String 0 (high e, printed on top) gets the highest y."""
        return NUM_STRINGS - 1 - string_index

    def draw_static_fretboard(self) -> None:
        """Draws the neck: 6 string lines, fret wires (nut bolded), fret
        number labels, and string name labels. Safe to call more than
        once (clears and redraws)."""
        self.ax.clear()
        self._fret_texts = []
        self._string_texts = []

        for string_index in range(NUM_STRINGS):
            y = self._string_y(string_index)
            self.ax.plot(
                [self._fret_wire_x(0) - 0.02, self._fret_wire_x(self.num_frets)],
                [y, y],
                color="#8a8a8a",
                linewidth=1,
                zorder=1,
            )
            string_text = self.ax.text(
                self._fret_wire_x(0) - 0.03,
                y,
                STRING_NAMES_TOP_TO_BOTTOM[string_index],
                ha="right",
                va="center",
                fontsize=10,
                fontweight="bold",
            )
            self._string_texts.append((string_text, 10))

        for fret in range(self.num_frets + 1):
            x = self._fret_wire_x(fret)
            self.ax.plot(
                [x, x],
                [-0.5, NUM_STRINGS - 0.5],
                color="black" if fret == 0 else "#bbbbbb",
                linewidth=4 if fret == 0 else 1,
                zorder=1,
            )
            if fret > 0:
                fret_text = self.ax.text(
                    x - (x - self._fret_wire_x(fret - 1)) / 2,
                    -0.85,
                    str(fret),
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="#666666",
                )
                self._fret_texts.append((fret_text, 8))

        self.ax.set_xlim(self._fret_wire_x(0) - 0.06, self._fret_wire_x(self.num_frets) + 0.01)
        self.ax.set_ylim(-1.1, NUM_STRINGS - 0.3)
        self.ax.axis("off")
        self._draw_legend()
        self._board_drawn = True
        self._on_resize()  # apply the current window size's scale immediately, not just on the next resize

    def _draw_legend(self, fontsize: float = 8) -> None:
        """Which color/marker means which finger -- baked into the
        figure itself (not just described in a README), so it's still
        readable in an exported gif with no surrounding context."""
        handles = [
            plt.Line2D(
                [], [], marker="o", linestyle="", markersize=9,
                markerfacecolor=FINGER_COLORS[finger], markeredgecolor="black",
                label=FINGER_LABELS[finger],
            )
            for finger in (1, 2, 3, 4, None)
        ]
        handles.append(
            plt.Line2D(
                [], [], marker="x", linestyle="", markersize=9,
                markeredgecolor="#333333", markeredgewidth=2, label="muted (x)",
            )
        )
        self._legend = self.ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
            ncol=6,
            fontsize=fontsize,
            frameon=False,
            handletextpad=0.3,
            columnspacing=1.0,
        )

    def _ensure_frets_cover(self, moments: list[Moment]) -> None:
        """If the song actually uses frets beyond what this instance was
        configured to draw, extend rather than silently clipping notes
        off the edge of the neck."""
        max_fret = 0
        for m in moments:
            for n in m.notes:
                if n.fret and n.fret > max_fret:
                    max_fret = n.fret
        if max_fret > self.num_frets:
            print(
                f"NOTE: song uses fret {max_fret}, beyond the configured "
                f"num_frets={self.num_frets} -- extending the drawn neck to fit."
            )
            self.num_frets = max_fret

    def _bend_target_fret(self, note) -> float:
        """Where a bend/rebend/release note's pitch is headed, in fret
        (== semitone) units. Falls back to a whole-step (2 frets) when
        no explicit target/amount is given in the tab -- matches this
        project's established reading of an unqualified '^' as a
        full/whole-step bend. Used only to size the bend's visual
        amplitude now, not to move the dot along the neck (see
        `_note_offset_at`)."""
        if note.technique_target is not None:
            return note.technique_target
        if note.bend_amount is not None:
            return note.fret + note.bend_amount * 2
        return note.fret + 2

    def _bend_amplitude(self, note) -> float:
        """How far (in string-spacing units) a bend's dot deflects off
        its string line. Bigger bend -> bigger push, capped so it never
        visually reaches the neighboring string's own line."""
        semitones = abs(self._bend_target_fret(note) - note.fret)
        return min(0.6, 0.12 * max(semitones, 1))

    def _note_offset_at(self, note, t: float, moment: Moment) -> tuple[float, float]:
        """
        (dx, dy) displacement from a note's plain (fret, string)
        position at time `t`. A real string bend deflects the string
        SIDEWAYS relative to its own length -- toward a neighboring
        string, not up the neck to a different fret -- so this only
        ever moves the dot vertically (dy); dx is always 0. Direction
        is toward the neighboring string closer to the high e string
        (the standard way a bend is physically pushed), except on the
        high e string itself, which has no string above it to push
        into and bends the other way, toward B.
        """
        if moment.duration and moment.duration > 0:
            progress = min(1.0, max(0.0, (t - moment.start_time) / moment.duration))
        else:
            progress = 1.0

        direction = -1.0 if note.string_index == 0 else 1.0

        if note.modifier == Technique.PRE_BEND:
            # Already bent before being picked -- static for the whole note.
            return 0.0, direction * self._bend_amplitude(note)

        if note.modifier in BEND_TECHNIQUES:
            amplitude = self._bend_amplitude(note)
            # Ramp over the first 40% of the hold, then stay deflected.
            ramp = min(1.0, progress / 0.4)
            if note.modifier == Technique.RELEASE:
                # A release is undoing an existing bend -- starts
                # deflected, ramps back down to the string's rest line.
                ramp = 1.0 - ramp
            return 0.0, direction * amplitude * ramp

        if note.modifier in VIBRATO_TECHNIQUES:
            wiggle = 0.18 * math.sin(2 * math.pi * 6.0 * t)
            return 0.0, wiggle

        return 0.0, 0.0

    def animate(
        self,
        moments: list[Moment],
        assignments: list[list[HandPosition]],
        fps: int = 30,
    ) -> FuncAnimation:
        """
        Per frame, draws one dot per HandPosition in the current moment's
        assignment -- chords draw multiple dots naturally, no special
        casing needed here since that decision already happened
        upstream in the optimizer.
        """
        if not moments:
            raise ValueError("no moments to animate")
        if len(moments) != len(assignments):
            raise ValueError(
                f"moments ({len(moments)}) and assignments ({len(assignments)}) "
                f"must be the same length and same order -- pass optimizer.solve()'s "
                f"output directly."
            )
        for m in moments:
            if m.start_time is None or m.duration is None:
                raise ValueError(
                    f"Moment at column {m.column} has no start_time/duration -- "
                    f"run rhythm.assign_timing() before animate()."
                )

        self._ensure_frets_cover(moments)
        if not self._board_drawn:
            self.draw_static_fretboard()

        total_duration = moments[-1].start_time + moments[-1].duration
        self._fps = fps
        n_frames = max(1, int(math.ceil(total_duration * fps)))

        render = self._make_frame_renderer(moments, assignments)

        def update(frame_idx):
            return render(frame_idx / fps)

        return FuncAnimation(
            self.fig, update, frames=n_frames, interval=1000 / fps, blit=False
        )

    def _make_frame_renderer(self, moments: list[Moment], assignments: list[list[HandPosition]]):
        """
        Builds the scatter/text artists once and returns a `render(t)`
        closure that moves them to time `t`'s state and returns the
        updated artist tuple. Shared by `animate()` (driven by
        FuncAnimation's frame clock) and `show_interactive()` (driven by
        a scrub slider) so both play back identically -- one rendering
        path, two ways of choosing `t`.
        """
        start_times = [m.start_time for m in moments]

        fretted_scatter = self.ax.scatter([], [], s=160, zorder=5, edgecolors="black", linewidths=0.8)
        muted_scatter = self.ax.scatter([], [], s=160, marker="x", zorder=5, c="#333333", linewidths=2)
        time_text = self.ax.text(
            0.99, 1.02, "", transform=self.ax.transAxes, ha="right", va="bottom", fontsize=9
        )
        # Minimal stand-in for "where to put your wrist": just the
        # current moment's anchor fret as a number, not a drawn
        # span/bracket -- one clearly-labeled value, updated per moment.
        anchor_text = self.ax.text(
            0.01, 1.02, "", transform=self.ax.transAxes, ha="left", va="bottom", fontsize=9
        )
        self._time_text = time_text
        self._anchor_text = anchor_text
        self._on_resize()  # these new texts start at their base size -- apply the current window scale to them too

        def render(t: float):
            idx = bisect.bisect_right(start_times, t) - 1
            idx = max(0, min(idx, len(moments) - 1))
            moment = moments[idx]
            assignment = assignments[idx]

            anchors = [hp.anchor_fret for hp in assignment if hp.anchor_fret is not None]
            anchor_text.set_text(f"Anchor: fret {min(anchors)}" if anchors else "Anchor: --")

            fretted_xy, fretted_colors = [], []
            muted_xy = []
            for hp in assignment:
                note = hp.note
                base_y = self._string_y(note.string_index)
                if note.fret < 0:
                    # dead/muted note ("x")
                    muted_xy.append((self._fret_x(0), base_y))
                    continue
                dx, dy = self._note_offset_at(note, t, moment)
                fretted_xy.append((self._fret_x(note.fret) + dx, base_y + dy))
                fretted_colors.append(FINGER_COLORS.get(hp.finger, FINGER_COLORS[None]))

            fretted_scatter.set_offsets(np.array(fretted_xy) if fretted_xy else np.empty((0, 2)))
            if fretted_colors:
                fretted_scatter.set_facecolor(fretted_colors)
            muted_scatter.set_offsets(np.array(muted_xy) if muted_xy else np.empty((0, 2)))
            time_text.set_text(f"t = {t:5.2f}s")
            return fretted_scatter, muted_scatter, time_text, anchor_text

        return render

    def show_interactive(
        self,
        moments: list[Moment],
        assignments: list[list[HandPosition]],
        fps: int = 30,
        initial_speed: float = 1.0,
        audio_track=None,
    ) -> None:
        """
        Opens a live matplotlib window with a scrub slider, a speed
        slider, a play/pause button, and (if `audio_track` is given) a
        Resync Audio button -- still a local script (`plt.show()`
        blocks until the window is closed), no web frontend involved.
        Dragging the time slider scrubs to any point instantly; Play
        advances it automatically using the same per-frame rendering
        `animate()` uses, so scrubbing and playback always agree on
        what a given moment in time looks like.

        TIMING IS WALL-CLOCK BASED, NOT FRAME-COUNTED
        --------------------------------------------------
        Each `advance()` tick computes `t` from REAL elapsed time
        (`time.perf_counter()`) since playback started, not by adding a
        fixed `1/fps` every tick. A fixed-per-tick approach silently
        assumes the timer fires at EXACTLY `fps` times a second, every
        time -- if a frame's render work (or Tkinter's timer itself)
        takes even a little longer than that on a dense passage, the
        picture falls behind real time and drifts out of sync with
        audio (which keeps playing at real speed regardless). Deriving
        `t` from the actual clock instead means a slow frame just
        results in one bigger jump forward next tick, not permanent
        drift -- this is what actually keeps the dots aligned with the
        audio.

        The speed slider (0.1x-4x, default `initial_speed`) scales how
        fast simulated song-time advances per real second WITHOUT
        changing `fps` -- the render itself always redraws at `fps`
        wall-clock frames/second, only the size of each simulated-time
        step changes.

        `audio_track` (optional): the full-song numpy array from
        `src/audio.render_track()` for this exact `moments`/
        `assignments` pair (NOT a file path -- kept in memory so an
        arbitrary slice of it can be written out and played on demand).
        `winsound` can only play a wav from its own beginning and has
        no pitch-preserving variable-rate playback, so:
          - audio (re)starts, sliced from the CURRENT scrub position,
            every time Play is pressed WHILE the speed slider reads
            1.0x -- this is what lets Play correctly RESUME audio after
            a pause or a scrub, instead of it being gone for good.
          - dragging the speed slider away from 1.0x stops audio (can't
            play it at the wrong rate); dragging it back to exactly
            1.0x while still playing automatically restarts audio from
            the current position.
          - the Resync Audio button is the explicit manual version of
            the same thing: snaps speed back to 1.0x and restarts audio
            from wherever the picture currently is.
          - at any other speed, audio stays silent rather than playing
            something misleadingly wrong.
        """
        from matplotlib.widgets import Button, Slider

        from src import audio as audio_module

        if not moments:
            raise ValueError("no moments to animate")
        if len(moments) != len(assignments):
            raise ValueError(
                f"moments ({len(moments)}) and assignments ({len(assignments)}) "
                f"must be the same length and same order -- pass optimizer.solve()'s "
                f"output directly."
            )
        for m in moments:
            if m.start_time is None or m.duration is None:
                raise ValueError(
                    f"Moment at column {m.column} has no start_time/duration -- "
                    f"run rhythm.assign_timing() before show_interactive()."
                )

        self._ensure_frets_cover(moments)
        self.draw_static_fretboard()  # always fresh: leaves room for the widgets below

        total_duration = moments[-1].start_time + moments[-1].duration
        render = self._make_frame_renderer(moments, assignments)

        self.fig.subplots_adjust(bottom=0.36)
        slider_ax = self.fig.add_axes((0.13, 0.20, 0.55, 0.05))
        speed_ax = self.fig.add_axes((0.13, 0.10, 0.55, 0.05))
        button_ax = self.fig.add_axes((0.74, 0.155, 0.14, 0.09))
        resync_ax = self.fig.add_axes((0.74, 0.045, 0.14, 0.09))

        slider = Slider(slider_ax, "t (s)", 0.0, total_duration, valinit=0.0)
        speed_slider = Slider(
            speed_ax, "speed", 0.1, 4.0, valinit=initial_speed, valfmt="%.2fx"
        )
        play_button = Button(button_ax, "▶ Play")
        resync_button = Button(resync_ax, "🔊 Resync Audio")

        state = {
            "playing": False,
            "audio_playing": False,
            "advancing": False,
            # Wall-clock playback baseline: `t` at any moment while
            # playing is derived from these two, not accumulated tick
            # by tick -- see the wall-clock docstring above.
            "play_start_real": 0.0,
            "play_start_sim": 0.0,
        }

        def stop_audio_if_playing():
            if state["audio_playing"]:
                audio_module.stop_playback()
                state["audio_playing"] = False

        def rebase_clock(sim_t: float | None = None):
            """Resets the wall-clock baseline to "right now = `sim_t`"
            -- called whenever playback starts, the speed changes, or a
            manual scrub happens, so the next `advance()` tick computes
            forward from the right starting point instead of jumping."""
            state["play_start_real"] = time.perf_counter()
            state["play_start_sim"] = slider.val if sim_t is None else sim_t

        def play_audio_from(t: float):
            """(Re)starts audio playback sliced from song-time `t`
            onward -- works from ANY position, not just the start, so
            Play correctly RESUMES audio after a pause/scrub instead of
            it staying gone for good."""
            if audio_track is None:
                return
            start_sample = max(0, int(t * getattr(audio_module, "SAMPLE_RATE", 44100)))
            segment = audio_track[start_sample:]
            if len(segment) == 0:
                return
            stop_audio_if_playing()
            temp_path = audio_module.write_temp_wav(segment)
            audio_module.play_wav_async(temp_path)
            state["audio_playing"] = True

        def on_slider_change(val):
            render(val)
            self.fig.canvas.draw_idle()
            # A manual scrub (dragging the slider by hand) while audio
            # is running would leave the sound playing from wherever it
            # already was, no longer matching what's on screen -- stop
            # it and rebase the clock rather than let it drift into
            # obvious nonsense. BUT normal playback ALSO moves this
            # slider every frame (see `advance()` below), which fires
            # this exact same callback -- without the
            # `state["advancing"]` guard, that reads as a "scrub" on
            # every single frame and kills the audio within about one
            # frame of starting it, every time. Only a move NOT caused
            # by `advance()` counts as a real scrub.
            if state["playing"] and not state["advancing"]:
                stop_audio_if_playing()
                rebase_clock(val)

        slider.on_changed(on_slider_change)

        def on_speed_change(val):
            stop_audio_if_playing()
            rebase_clock()
            # The one explicit thing the user asked for: restoring
            # speed to exactly 1.0x while still playing brings audio
            # back automatically, no extra click needed.
            if state["playing"] and val == 1.0:
                play_audio_from(slider.val)

        speed_slider.on_changed(on_speed_change)

        timer = self.fig.canvas.new_timer(interval=1000 / fps)

        def advance():
            if not state["playing"]:
                return
            elapsed_real = time.perf_counter() - state["play_start_real"]
            next_t = state["play_start_sim"] + speed_slider.val * elapsed_real
            state["advancing"] = True
            try:
                if next_t >= total_duration:
                    slider.set_val(total_duration)
                    state["playing"] = False
                    play_button.label.set_text("▶ Play")
                    timer.stop()
                    stop_audio_if_playing()
                    return
                slider.set_val(next_t)
            finally:
                state["advancing"] = False

        timer.add_callback(advance)
        timer.stop()  # started for real by on_play_clicked, once audio's had time to catch up

        def on_close(_event):
            # Without this, a timer callback can fire after the window
            # (and its underlying Tk widget) is already gone, which
            # throws an ugly "invalid command name ... on_timer" error
            # on exit instead of closing cleanly.
            timer.stop()
            stop_audio_if_playing()

        self.fig.canvas.mpl_connect("close_event", on_close)

        # A one-shot delay timer: gives the audio device a moment to
        # actually start producing sound before the animation clock
        # starts advancing -- see audio.PLAYBACK_LATENCY_SECONDS'
        # docstring for why this gap is real and how to retune it. The
        # wall-clock baseline is rebased right when the timer actually
        # starts ticking (not when Play was clicked), so this delay
        # doesn't itself show up as a jump in the picture.
        start_delay_timer = self.fig.canvas.new_timer(
            interval=int(audio_module.PLAYBACK_LATENCY_SECONDS * 1000)
        )
        start_delay_timer.single_shot = True

        def begin_advancing():
            if state["playing"]:
                rebase_clock()
                timer.start()

        start_delay_timer.add_callback(begin_advancing)

        def on_play_clicked(_event):
            if state["playing"]:
                state["playing"] = False
                play_button.label.set_text("▶ Play")
                timer.stop()
                stop_audio_if_playing()
            else:
                if slider.val >= total_duration:
                    slider.set_val(0.0)  # restart from the beginning if replaying at the end
                state["playing"] = True
                play_button.label.set_text("⏸ Pause")
                if audio_track is not None and speed_slider.val == 1.0:
                    play_audio_from(slider.val)
                    start_delay_timer.start()  # let audio actually start before the picture does
                else:
                    begin_advancing()  # no audio to wait for -- start advancing right away

        play_button.on_clicked(on_play_clicked)

        def on_resync_clicked(_event):
            # The explicit manual escape hatch: whatever state audio
            # got left in (silent after a scrub, wrong speed, etc.),
            # this snaps speed back to 1.0x and restarts audio in sync
            # with wherever the picture currently is.
            if speed_slider.val != 1.0:
                speed_slider.set_val(1.0)  # on_speed_change auto-restarts audio if still playing
            elif state["playing"]:
                play_audio_from(slider.val)
                rebase_clock()

        resync_button.on_clicked(on_resync_clicked)

        render(0.0)
        # Keep references alive -- matplotlib widgets stop responding if
        # their controller objects get garbage collected.
        self._interactive_widgets = (
            slider, speed_slider, play_button, resync_button, timer, start_delay_timer,
        )

        plt.show()

    def save(self, animation: FuncAnimation, path: str) -> None:
        """
        Writes the animation to disk. GIF (via the 'pillow' writer,
        always available since it's a matplotlib dependency) unless the
        path asks for something else -- ffmpeg isn't guaranteed to be
        installed on a given machine, so .gif is the safe default this
        project can actually produce everywhere.
        """
        fps = getattr(self, "_fps", 30)
        if path.lower().endswith(".gif"):
            animation.save(path, writer="pillow", fps=fps)
        else:
            animation.save(path, writer="ffmpeg", fps=fps)


if __name__ == "__main__":
    # Small end-to-end demo: parse a real section of Hotel California,
    # group into moments, time it, solve fingering, then either save a
    # gif (default) or open the live interactive viewer:
    #   python -m src.visualizer              -> saves hotel_california_demo.gif
    #   python -m src.visualizer --interactive -> opens a window with a
    #                                              scrub slider + play/pause
    # (--interactive isn't the default because plt.show() blocks until
    # the window is closed, which would hang an automated/non-GUI run.)
    import sys

    from src.parser import parse_tab
    from src.moments import group_into_moments
    from src.rhythm import assign_timing
    from src.optimizer import FingeringOptimizer

    with open("data/hotel_california_solo.txt", encoding="utf-8") as f:
        raw = f.read()

    # Just the first section (the "(4:20)" block) for a quick, watchable demo.
    lines = raw.split("\n")
    section = "\n".join(lines[5:11])

    notes = parse_tab(section)
    # Hotel California is 75 BPM -- see rhythm.py for how bpm here maps
    # to seconds_per_column, and how that mapping was calibrated against
    # this exact song's own embedded recording timestamps.
    moments = assign_timing(group_into_moments(notes), bpm=75)
    solution = FingeringOptimizer().solve(moments)

    viz = FretboardVisualizer(num_frets=16)

    if "--interactive" in sys.argv:
        viz.show_interactive(moments, solution, fps=20)
    else:
        anim = viz.animate(moments, solution, fps=20)
        viz.save(anim, "hotel_california_demo.gif")
        print("Saved hotel_california_demo.gif")
