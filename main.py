"""
main.py

Ties the whole pipeline together, per PLAN.md:
    raw tab text -> parser -> moments -> rhythm -> optimizer -> visualizer (+ audio)

USAGE
------
Run it with NO arguments and a small picker window opens first --
choose a tab file and (optionally) type in the song's BPM there, no
terminal typing needed. Clicking Start closes that window and opens
the real animated fretboard in its own separate window:
    python main.py

Or skip the picker entirely by passing a tab file directly (useful for
scripted/repeated runs) -- opens the interactive window right away:
    python main.py data/hotel_california_solo.txt --bpm 75

Pass --bpm if you know the song's real tempo -- it meaningfully changes
playback timing (see rhythm.py). BPM has to be known before the
optimizer and timing are computed at all, and it can't be changed
mid-run without re-running the whole pipeline, so there's no live
control for it once the visualizer window is open (the picker window,
or --bpm, is the only place to set it).

To save a gif (+ a companion .wav you can play alongside it) instead of
opening a window:
    python main.py data/hotel_california_solo.txt --bpm 75 --gif

Every run (gif or interactive) also writes reports/<tab name>_fingering.txt --
a plain-text, one-line-per-note list of the whole solved fingering
(string, fret, technique, chosen finger, anchor fret), for checking it
by hand against a real tab/reference (see src/report.py). The reports/
folder is created automatically if it doesn't exist yet.
"""

import argparse
import os
import sys

from src.audio import render_track, write_wav
from src.moments import group_into_moments
from src.optimizer import FingeringOptimizer
from src.parser import parse_tab
from src.report import next_report_path, write_fingering_report
from src.rhythm import assign_timing
from src.visualizer import FretboardVisualizer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse a plain-text guitar tab, solve a fingering, and animate + play it on a fretboard."
    )
    parser.add_argument(
        "tab_file", nargs="?", default=None,
        help="path to a plain-text tab file, e.g. data/hotel_california_solo.txt -- "
             "omit this entirely to get a picker window instead",
    )
    parser.add_argument(
        "--bpm", type=float, default=None,
        help="the song's real tempo, if you know it. Used to derive realistic playback timing (see rhythm.py). "
             "If omitted, falls back to a rougher default timing guess (no interactive prompt).",
    )
    parser.add_argument(
        "--columns-per-beat", type=float, default=8.0,
        help="only used with --bpm: how many tab columns represent one beat (default 8, calibrated against Hotel California -- see rhythm.py)",
    )
    parser.add_argument(
        "--seconds-per-column", type=float, default=0.05,
        help="only used WITHOUT --bpm: raw column-to-seconds scale (default 0.05)",
    )
    parser.add_argument(
        "--tolerance", type=int, default=1,
        help="column tolerance for grouping notes into a shared moment/chord (default 1 -- see moments.py)",
    )
    # default=None, not a hardcoded number: FingeringOptimizer's own
    # constructor defaults (src/optimizer.py) are the single source of
    # truth for "what does the model believe by default" -- that's
    # where these get tuned by ear. A second hardcoded default here
    # would silently shadow every tuning change made there unless this
    # file was ALSO updated to match, which is exactly what happened
    # (shift_weight was bumped to 2.5 in optimizer.py, but this flag's
    # default stayed at 1.0 and kept winning on every run that didn't
    # pass --shift-weight explicitly). None means "the user didn't ask
    # for an override" -- see the FingeringOptimizer(...) call below,
    # which only passes a keyword through when the flag was actually
    # set, letting FingeringOptimizer's own default apply otherwise.
    parser.add_argument("--stretch-weight", type=float, default=None, help="optimizer: cost per unit of within-chord finger spread (default: FingeringOptimizer's own)")
    parser.add_argument("--shift-weight", type=float, default=None, help="optimizer: cost per unit of hand travel between moments (default: FingeringOptimizer's own)")
    parser.add_argument("--max-stretch-frets", type=int, default=None, help="optimizer: max comfortable one-hand fret span (default: FingeringOptimizer's own)")
    parser.add_argument("--num-frets", type=int, default=15, help="how many frets to draw on the fretboard (auto-extends if the song needs more)")
    parser.add_argument("--fps", type=int, default=20, help="playback frame rate")
    parser.add_argument(
        "--gif", action="store_true",
        help="save a gif (+ companion .wav) instead of opening the live interactive window",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="(default behavior already -- kept as a no-op flag for backward compatibility)",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="only used in the interactive window: starting playback speed multiplier (0.1-4.0, default 1.0 -- can also be dragged live in the window)",
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="skip synthesizing/playing audio entirely (see src/audio.py)",
    )
    parser.add_argument(
        "--output", default=None,
        help="gif output path (default: <tab_file's name>_fingering.gif); only used with --gif",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    tab_file = args.tab_file
    bpm = args.bpm

    if tab_file is None:
        from src.launcher import prompt_for_inputs

        picked = prompt_for_inputs()
        if picked is None:
            print("Cancelled -- no tab file chosen.")
            return
        tab_file, picked_bpm = picked
        if bpm is None:
            bpm = picked_bpm

    if bpm is None:
        print(
            "No BPM given -- using the rough default column-based timing "
            "(pass --bpm <number>, or type one into the picker window, for accurate playback speed)."
        )

    with open(tab_file, encoding="utf-8") as f:
        raw = f.read()

    notes = parse_tab(raw)
    moments = group_into_moments(notes, tolerance=args.tolerance)
    if bpm is not None:
        moments = assign_timing(moments, bpm=bpm, columns_per_beat=args.columns_per_beat)
    else:
        moments = assign_timing(moments, seconds_per_column=args.seconds_per_column)

    # Only pass through the args the user actually typed on the command
    # line -- an unset (None) flag means "no override," so
    # FingeringOptimizer's own constructor default applies untouched,
    # instead of this file silently reintroducing a second, competing
    # default that can drift out of sync with the real one.
    overrides = {
        "stretch_weight": args.stretch_weight,
        "shift_weight": args.shift_weight,
        "max_stretch_frets": args.max_stretch_frets,
    }
    optimizer = FingeringOptimizer(**{k: v for k, v in overrides.items() if v is not None})
    solution = optimizer.solve(moments)

    viz = FretboardVisualizer(num_frets=args.num_frets)

    stem = os.path.splitext(os.path.basename(tab_file))[0]

    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    report_path = next_report_path(reports_dir, stem)
    write_fingering_report(report_path, moments, solution)
    print(f"Saved {report_path} (full fingering list, {len(moments)} moments -- for checking accuracy by hand)")

    audio_path = None
    track = None
    if not args.no_audio:
        track = render_track(moments, solution)
        audio_path = f"{stem}_audio.wav"
        write_wav(audio_path, track)
        print(f"Saved {audio_path}")

    if args.gif:
        output_path = args.output or f"{stem}_fingering.gif"
        anim = viz.animate(moments, solution, fps=args.fps)
        viz.save(anim, output_path)
        print(f"Saved {output_path} ({len(moments)} moments)")
        if audio_path:
            print(f"(gifs can't hold audio -- play {audio_path} alongside it manually if you want sound)")
        return

    viz.show_interactive(moments, solution, fps=args.fps, initial_speed=args.speed, audio_track=track)


if __name__ == "__main__":
    main(sys.argv[1:])
