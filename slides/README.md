# Slides

One-slide presentations of the system. Each folder holds:

| File | What it is |
|---|---|
| `slide.png` | The slide as an image (1920×1080) |
| `slide.pdf` | The slide as a PDF page |
| `slide.html` | The slide as a web page; open it in any browser |
| `speaker-notes.md` | What to say, and answers to likely questions |
| `img/` | The step pictures, produced by running the system on a real scan |

| Folder | Content |
|---|---|
| [`01-introduction`](01-introduction/) | What the system is, what it uses, how it avoids AI, and its 7 steps |
| [`02-how-it-works`](02-how-it-works/) | How it works, including the fluid step: 8 steps on a real test-patient scan, the solid tumor in orange and its fluid in cyan |

The step pictures come from the system itself (`python main.py detect ...`). A
slide is re-exported with `scripts/export_slide.py`.
