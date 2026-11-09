# Speaker notes: Finding the tumor and the fluid inside it

OPENING: This slide shows how the system works, step by step, on a real scan from a test patient. The new part is step 6: finding the fluid inside a tumor.

WHY FLUID MATTERS: Many tumors, especially gliomas, contain fluid: cysts or dead tissue. On this kind of MRI that fluid is dark, as dark as the normal fluid around the brain. A system that only looks for bright tissue treats it as background and misses most of the tumor. In this dataset most gliomas are largely dark.

THE STEPS:
1. Load the scan as grayscale, brightness rescaled to 0–1.
2. Denoise: a median filter takes the middle value of each pixel’s neighbours. Noise goes, edges stay sharp.
3. Enhance: CLAHE boosts contrast in small tiles so faint regions stand out.
4. Remove the skull: a threshold finds the head, erosion shrinks it inward past the skull.
5. Bright areas: Otsu’s method picks the brightness cut-offs that best separate the tissues; the brightest group is the solid, enhancing tumor candidate.
6. Fluid areas (new): the darkest group is fluid. A dark area counts as a fluid-filled tumor only if it is round and solid in shape, thick (not a groove between brain folds), inside the brain and off its edge, and NOT mirrored on the other side of the head. Normal fluid chambers (ventricles) come in mirrored pairs or sit on the midline; a tumor is usually on one side.
7. Pick the tumor: the thickest, most tumor-like candidate wins. If the bright part forms the wall of the fluid area, both together are the tumor. Here: orange = solid part, cyan = fluid. 98% of the cyan area lies inside the doctor’s outline.
8. Suggest the type with written rules: pituitary tumors sit at the centre and are small; meningiomas are bright, even and near the skull; gliomas are patchy, inside the brain, and often largely dark.

IF ASKED:
- Isn’t the mirror check a kind of learning? No. It flips the region across the head’s midline and measures how much of the flipped region is also dark. A fixed calculation, the same for every scan.
- Did the fluid step help? Tested on 1,843 slices from patients the rules never saw: the tumor is found in the right place in 50.5% of slices (46% before); gliomas, the tumors that are mostly fluid, went from 29% to 40%. The type is right 71% of the time when the tumor is found (64% before), and 82.5% when the rules get the doctor’s outline.
- Does it create false alarms on healthy brains? Checked on 22 healthy slices: still 2 false alarms, the same as before the fluid step.
- What is still hard? Finding the tumor: about half the time it marks the wrong place. Meningiomas at the skull base are often taken for pituitary tumors.
- Is this a diagnosis? No. It marks a suspicious region and suggests a type.
