# Guitar Tab Fingering Visualizer

Take a plain-text guitar tab (from sites like Ultimate Guitar) and figure out which finger should play every note, then animate it on a fretboard.

I built this after getting annoyed that tab sites just show you fret numbers and leave the actual hand mechanics up to you. This is not that friendly for beginners, so this project uses an optimization algorithm to see which fingering combination is the best.

## Why this is more than a web toy

This turns out to be the same shape of problem as a bunch of classic algorithms: it's a shortest-path problem over a layered graph, computed with dynamic programming, and the structure is basically identical to the Viterbi algorithm used in speech recognition and HMMs. Instead of "most likely sequence of hidden states," it's "least effortful sequence of hand shapes." Each moment in the song is a layer; each possible hand position for that moment is a node; the edges are priced by how physically annoying that transition is.

## How it actually works

The pipeline is four steps, each one a separate file:

**1. Parser**: reads the raw tab text and turns it into a flat list of notes: which string, which fret, and what technique (bend, slide, hammer-on, vibrato, etc.) got used to play it. Tabs are inconsistent and messy in practice, such as multi-digit frets with no separator, missing string labels, bends written two different ways depending on who transcribed it. So, most of the real work here was catching and fixing those edge cases against an actual tab (Hotel California's solo), not writing the happy-path parser.

**2. Moments**: groups notes that happen at the same instant. A single note is its own moment; a two-note chord is one moment with two notes in it. This matters because a chord needs one joint fingering decision (you can't use your ring finger twice at once), not two independent ones.

**3. Rhythm**: plain text tabs don't actually encode timing, so this estimates how long each note is held based on how much space (dashes) comes before the next note. It's explicitly an approximation, not real transcription. I calibrated the timing constant against Hotel California's actual recording timestamps rather than just guessing a number, but it's still a heuristic and I'm not pretending otherwise.

**4. Optimizer**: the actual algorithm. For every moment, it works out every physically possible way to finger it (respecting hand span, barre chords, one-finger-one-fret), then runs a dynamic program over the whole song to find the cheapest path through all of them. "Cheapest" is priced on three things:
- how far your fingers have to stretch within one chord
- how far your whole hand has to move between one moment and the next (discounted if you've got more time to make the move)
- which finger you're using, independent of distance — because without this, the optimizer will happily use your pinky to reach a note just to avoid moving your hand, which isn't how anyone actually plays

All of the distance math accounts for the fact that frets get physically narrower the higher up the neck you go: a two-fret stretch at fret 2 is a real reach, the same two frets at fret 15 is nothing. Treating every fret gap as equal-sized would have thrown off both the stretch and shift costs.

**5. Visualizer** — takes the optimizer's output and animates it on a fretboard, showing which finger lands where as the song plays.

## About the audio 

- Created note pitches: those give you a sample for an arbitrary IN-PROGRESS BEND (a continuously sliding pitch between two frets), which is exactly the
technique this project cares about animating. A dataset only has
fixed, static pitches -- making a bend sound right would still mean
pitch-shifting/synthesizing on top of the sampled audio, plus adding a
real network-fetch + licensing + multi-hundred-MB download dependency
for the non-bend notes you'd cover. A few dozen lines of signal
processing does the whole job honestly and offline instead. If bends
weren't the point of this project, EGFxSet in particular would be a
genuinely good swap-in -- worth revisiting if that tradeoff ever
changes.
So instead: every note is synthesized here, in Python, from its
(string, fret, technique) -- same inputs the visualizer already reads.
Bends/vibrato bend the actual synthesized pitch, not just the dot.



## What it doesn't do 

- Timing is approximate, not a real transcription. See the Rhythm section above.
- No audio synthesis of in-progress bends yet: sample libraries exist for clean notes, but none of them record a bend's continuous pitch slide, so that piece still needs to be built rather than sourced.
- Finger preference is currently a flat cost, which means it can still lose to a physically tiny hand-shift up near the highest frets. I found this while testing and I'm treating it as a known limitation rather than hiding it — a real fix means the finger-preference cost should scale with the local fret spacing instead of being constant, which I haven't built yet.

## Installation

```bash
git clone https://github.com/rh00000/guitar-tab-visualizer
cd guitar-tab-visualizer
pip install -r requirements.txt
```

## Usage

Drop a plain-text tab into `data/`, then run:

```bash
python main.py data/your_tab.txt
```

This runs the whole pipeline — parse, group into moments, assign timing, optimize fingering — and hands the result to the visualizer.

To just run a single piece of the pipeline (useful for debugging), each module works standalone too:

```bash
python -m src.parser data/your_tab.txt
python -m src.optimizer
```

## Examples

*(drop a screenshot or a short screen recording of the visualizer here once you've got one — this is the section people actually look at first)*

## Tech stack

Python, numpy, matplotlib, networkx.