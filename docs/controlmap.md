# Control maps: control-map extraction

`thenoise` can extract **control maps** from a plain image — the pictures that
ControlNet-style pipelines condition on. It works exactly like `/upscale`: image in, map
out, **no diffusion model loaded**. A map is always the size of the image it was made
from.

Five kinds ship today: **canny**, **lineart** and **softedge** need nothing extra, while
**depth** and **pose** need a small download each. Every one of them is available from
the CLI (`controlmap`), the web UI's Control Maps tab and the `/controlmap` endpoint, with
the same options.

| Input | Canny Edge | Depth map | Pose | Lineart | Soft-edged map | 
|--------|---------|---------|---------|---------|---------|
|<img width="100%" alt="thenoise-input-img" src="https://github.com/user-attachments/assets/50ce8ae9-3adf-457c-8822-d916a00219ca" /> | <img width="100%" alt="canny" src="https://github.com/user-attachments/assets/f210210d-b23e-481a-b373-bccd2efd1f8c" /> | <img width="100%" alt="depth" src="https://github.com/user-attachments/assets/62485b42-6863-43b7-8173-966e93408e82" /> | <img width="100%" alt="pose" src="https://github.com/user-attachments/assets/6a8aa67e-796d-45c2-8089-0830665c2991" /> | <img width="100%" alt="lineart" src="https://github.com/user-attachments/assets/d134594c-7a5c-4599-a132-3997fc84745b" /> | <img width="100%" alt="softedge" src="https://github.com/user-attachments/assets/a4157140-07f9-47e9-b2a7-f9cc0372998d" />|

## Installing and downloading what each kind needs

Canny, lineart and softedge work out of the box. Depth and pose need checkpoints:

```bash
python scripts/download.py --model controlmaps                # depth + pose + person detector
python scripts/download.py --model controlmaps --variant base  # depth, crisper
python scripts/download.py --model controlmaps --variant large # depth, crispest
python scripts/download.py --model controlmaps --variant pose       # pose + person detector
python scripts/download.py --model controlmaps --variant pose-r50vd # the better detector
```

Everything lands in `./models/controlmaps`, one directory per checkpoint. Point
`--controlmap-dir` at that directory and the engine works out which checkpoint is
which:

```bash
./thenoise.sh controlmap --input photo.jpg --type depth \
  --controlmap-dir ./models/controlmaps --out depth.png
./thenoise.sh controlmap --input photo.jpg --type pose \
  --controlmap-dir ./models/controlmaps --out pose.png
```

Pose needs **both** of its checkpoints: one finds the people, the other measures their
joints. With only one in the directory `pose` is not offered, and `--list` says what is
missing rather than staying quiet about it:

```bash
./thenoise.sh controlmap --list --controlmap-dir ./models/controlmaps
```

Ask for more than one `--type` and each map gets its own file (`map_canny.png`,
`map_pose.png`, …).

## The depth map

Relative depth — **bright is near, dark is far** — rather than distance in metres.

| Option | Default | Meaning |
|--------|---------|---------|
| `invert` | `false` | Swap near/far (dark = near) |
| `colormap` | `gray` | `gray`, or `turbo`: a vivid rainbow that is easier to read by eye |

## The pose map

An openpose-style skeleton — coloured bones and joints on a black canvas. It is *drawn*
rather than stretched, so a limb keeps its angle whatever size the map ends up. A joint
the model is unsure about is left out together with the bones hanging from it, so a
half-detected person loses that limb instead of gaining a stray line.

Bodies only: no hands, no face.

| Option | Default | Meaning |
|--------|---------|---------|
| `detector` | first found | Which person detector to use (see the variants above) |
| `threshold` | `0.5` | How sure the engine has to be that there is a person there at all |
| `kp_threshold` | `0.3` | How sure it has to be about a joint before drawing it |
| `line_width` | `4` | Bone thickness, in pixels |
| `radius` | `4` | Radius of the joint dots |
| `background` | `black` | `black`, or `green` — the canvas the original openpose maps used |

## The canny map

White edges on black: the classic hard-edged edge map.

| Option | Default | Meaning |
|--------|---------|---------|
| `low` | `100` | How faint an edge may be and still be drawn — lower finds more |
| `high` | `200` | Where an edge counts as strong; faint ones survive only where they run into a strong edge |
| `thickness` | `1` | Line thickness, in pixels |

## The lineart map

Thin dark lines on white: cleaner and more drawing-like than canny, with a single
threshold instead of two. An approximation of the model-based `lineart` annotator, with
nothing to download.

| Option | Default | Meaning |
|--------|---------|---------|
| `threshold` | `0.5` | `0`–`1`: how strong a line has to be to be drawn, measured against the strongest line in the image |
| `thickness` | `1` | Line thickness, in pixels |

## The softedge map

Edges as a soft grey gradient instead of a hard line — anti-aliased dark edges on white,
close to the `softedge` annotator, with nothing to download. Good when you want the
generation to keep soft shading rather than snap to outlines.

| Option | Default | Meaning |
|--------|---------|---------|
| `threshold` | `0.2` | `0`–`1`: where an edge starts to count |
| `blur` | `1.5` | Extra smoothing, in pixels; `0` leaves edges crisp |

## What the map remembers

Every map is saved with a `controlmap_data` chunk recording its type, the model that made
it and any option you changed, next to whatever metadata the input image already carried.
A map therefore says where it came from later on — on disk and in the web UI's info
dialog.
