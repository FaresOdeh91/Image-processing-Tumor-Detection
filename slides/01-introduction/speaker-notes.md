# Speaker notes: Brain tumor detection with image processing

OPENING: This is a system that detects brain tumors on MRI scans using image processing. It finds the tumor, outlines it, and suggests which of three types it is: glioma, meningioma or pituitary tumor.

WHAT IT USES: Python and three standard scientific libraries (NumPy, SciPy, scikit-image). The tools are classic image-processing operations: filters, brightness thresholds, shape operations (morphology) and measurements.

HOW IT AVOIDS AI: An AI system learns its own rules from thousands of labelled examples, and those rules are hidden inside a model. This system has no learning step. Every decision is a number measured on the image and compared with a rule a person wrote down. Nothing is stored between scans, so the same scan always gives the same answer, and every step can be shown on screen.

HOW IT WORKS:
1. Load: the scan is read as a grayscale image and its brightness is rescaled to 0-1, so scans from different machines are comparable.
2. Denoise: a median filter replaces each pixel with the middle value of its neighbours. Noise is removed; the tumor's edges stay sharp.
3. Enhance: CLAHE stretches contrast in small tiles, so a faint region stands out from the tissue around it.
4. Remove skull: the skull is bright and would look like a tumor. A threshold finds the head, then erosion (shrinking the shape) peels the skull away.
5. Find bright areas: on a contrast MRI the tumor lights up. Otsu's method picks the brightness cut-offs that best separate the tissue types, and the brightest group is kept.
6. Check rules: every bright blob is measured. Only one that is big enough, compact, clearly brighter than the brain, thick (not a thin blood vessel) and not touching the skull edge is kept.
7. Suggest type: rules from radiology give each type points. Pituitary tumors sit at the centre of the head and are small. Meningiomas are very bright and even, out near the skull. Gliomas are patchier and sit inside the brain tissue. Most points wins, and the reasons are shown.

IF ASKED:
- Is thresholding learning? No. Otsu computes the cut-off from this one image with a fixed formula, every time.
- How were the rules chosen? From radiology descriptions, checked on half the patients; the other half is kept for an untouched test.
- How accurate is it? Tested on patients the rules never saw (1,843 slices): it finds the tumor in the right place in 46% of slices. When it is given the expert's outline, the type rules are right 81% of the time. On 22 healthy slices it raised 2 false alarms. Finding the tumor is the hard part; the typing rules hold up.
- Is it a diagnosis? No. It points out a suspicious region and suggests a type.
- What can't it do? Tumors darker than normal brain are invisible to a brightness-based search.
