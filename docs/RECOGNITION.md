# Recognition stack

Implements the recognition layer and the LNN input of
[the software structure](ARCHITECTURE.md), sections 2E and 6.

## Why two networks

The CNN answers "what does this component look like right now". The LNN answers "what is this
component, given how it has looked and behaved while the board heated up". The plan is
explicit that the LNN must not consume high-resolution images: the detector locates
components, the CNN reduces each crop to a short embedding, and only that sequence reaches the
LNN. A five megapixel frame would otherwise dominate the temporal model and make training
impossible on bench hardware.

## CNN: `recognition/classification/cnn.py`

Two streams, RGB and 850 nm NIR, each a small convolutional encoder trained from scratch. No
pretrained weights are downloaded, so the stack works on an offline bench machine.

The streams stay separate rather than being concatenated at the input, because the plan feeds
visual features and NIR features to the LNN as distinct blocks. A failed illumination channel
then degrades one block instead of corrupting a single fused vector.

Outputs per crop: `rgb_embedding`, `nir_embedding`, and its own `class_logits` and
`package_logits`, which make the CNN trainable and evaluable on its own before the LNN exists.

## LNN: `recognition/lnn/cfc.py`

A closed-form continuous-time network. CfC is the closed-form solution of the liquid
time-constant ODE, so one step costs about as much as a GRU step while the state still evolves
in continuous time. One step:

```text
z      = backbone([x_t, h])
gate   = sigmoid(a(z) * dt + b(z))
h_next = (1 - gate) * f(z) + gate * g(z)
```

`dt` is the measured time since the previous frame set, taken from the capture timestamps.
This is the reason for choosing a liquid network over an LSTM here: an inspection is not
sampled on a fixed grid. The technician pauses, the shutter closes for a flat-field
correction, a frame set is rejected. A plain recurrent network would treat a 200 ms gap and a
40 s gap identically; this one does not. `test_elapsed_time_changes_the_state` pins that
behaviour down.

The hidden state can be carried between calls, so a live inspection can advance one frame set
at a time and get the same answer as a batch run over the whole sequence.

## Input vector

Per component and time step, as specified in the plan:

| Block | Width | Source |
|---|---|---|
| Visual features | CNN embedding | RGB stream |
| NIR features | CNN embedding | 850 nm stream |
| OCR features | 32 | Hashed character n-grams of the marking, plus confidence and a present flag |
| T_max, T_mean, dT, dT/dt | 4 | `processing/thermal/metrics.py` |
| Ambient temperature | 1 | Frame-set environment block |

`FeatureLayout` is the only place this order and these widths are defined, and every model
stores the layout it was built with, so a model cannot silently be fed a different vector.
Temperatures are scaled into roughly unit range for the network; the unscaled values stay in
the metrics objects for the interface and the reports.

`dT` is measured against the ambient temperature reported with the frame set. `dT/dt` needs
two captures, so the first step of a sequence reports zero rather than a guess.

## Outputs and rejection

The fusion model emits, for every time step, refined class logits, package logits,
thermal-condition logits and an unknown logit. Per step rather than only at the end, so the
interface can show confidence settling as the board heats.

`recognition/unknown_detection` then combines three independent signals: the learned unknown
head, the calibrated maximum class probability, and the normalized predictive entropy, which
catches two classes splitting the probability mass at a confidence that a threshold alone
would accept. Temperature scaling is a stored calibration value fitted on a validation set;
it defaults to 1.0, meaning uncalibrated, and a fitted value belongs in the registry entry
for the model.

## Deployment

`recognition/export.py` exports to ONNX with the dynamo exporter and the tests check the
outputs against ONNX Runtime.

Two forms of the fusion model are exported. The sequence form unrolls the time loop into the
graph, which fixes the number of steps and suits reprocessing a stored inspection. The step
form takes and returns the hidden state, which is what a live inspection needs. The tests
check that stepping reproduces the batched result.

Two export details cost real debugging time and are worth keeping in mind. The batch
dimension must be bounded, otherwise the exporter considers int64 max and the shapes derived
inside the LNN loop conflict. The example input must not have a batch of one, or the exporter
specializes that dimension and the exported graph silently accepts only single-item batches.

## Pretraining the RGB stream on public data

A downloaded picture library can train the CNN. It cannot train the LNN. The LNN's input is a
sequence with NIR and radiometric thermal terms, and no public dataset carries those channels
synchronized over time for PCB components, so every downloaded sequence would have length one
and a constant `dt`, which is exactly the case where the liquid dynamics contribute nothing.

`epr.apps.training_service` pretrains the RGB stream on the
[WACV 2019 PCB dataset](https://sites.google.com/view/chiawen-kuo/home/pcb-component-detection)
(Kuo et al., 47 board scans, PASCAL VOC annotations, direct download). FICS-PCB is larger but
requires a Trust-Hub account and a data agreement, so it is referenced but not automated.

Only the RGB stream is trained. The NIR stream stays randomly initialized until real 850 nm
captures exist. Feeding zeros to the NIR stream in order to reuse the whole model would teach
the fused head that NIR carries no information, which is the opposite of the design intent.

Three details in the loader are worth knowing, because each one silently inflates scores if
you get it wrong:

- **Split by physical board, not by scan.** The dataset ships boards as `ACM-109_Top` and
  `ACM-109_Bottom`, which are two sides of one PCB. Treating them as separate boards leaks the
  same illumination and component stock across the split. Merging them turns 47 scans into 34
  physical boards.
- **Labels carry designators and part numbers.** Annotations read `resistor R3`,
  `connector CNA`, `capacitor unknown` and `ic MK64FN1M0VLL12`. A trailing `unknown` means the
  designator was unreadable, not that the type is unknown, so stripping it recovers several
  hundred real components. Annotations whose leading token is `text`, `pins`, `pads` or `test`
  mark silkscreen and copper rather than parts, and are dropped. Cleaning takes 1028 raw label
  strings down to 14 usable classes.
- **Square the box before resizing.** Stretching an elongated chip resistor to a square makes
  it look like a square package, destroying a cue the classifier needs.

The class distribution is extremely skewed, roughly two thirds resistors and capacitors, so
training uses inverse-frequency class weights and balanced accuracy is the number to read.
Plain accuracy mostly measures the two big classes.

Current result on eight held-out boards: 0.714 accuracy and 0.650 balanced accuracy over 14
classes, against a 0.071 chance baseline.

Treat this as an initialization, not a finished classifier. The published images are about
560x701 pixels with a median component box near 42 pixels, while the bench head is a macro rig
at controlled distance and illumination. The domain gap is large enough that these weights
should be fine-tuned on real captures rather than trusted directly.

## Package prior from physical size

The plan's package classifier needs SOT, SOIC, QFP, BGA and the chip sizes. Those are
mechanical standards, not visual categories, and the frame set already reports the distance to
the board — so the size of a part can be measured rather than guessed.

`recognition/packages` reads the installed KiCad footprint libraries and builds a table of
package dimensions in millimetres. `processing/scale.py` converts a detected box to
millimetres using the reported distance. Nothing here is trained: the answer is a lookup
against published mechanical data, which is also what makes it explainable to a technician.

This attacks the failure the WACV authors named and that limits the CNN today. An 0603 and an
0805 chip resistor are nearly identical in appearance, but they are 1.6x0.8 mm and
2.0x1.25 mm. Measured against the table built from this machine's KiCad 9.0 install, 15188
footprints of which 14877 come from the true body outline:

| Measured | Top candidate | Score |
|---|---|---|
| 1.6 x 0.8 mm | 0603 | 1.000, and 0805 does not appear |
| 2.0 x 1.25 mm | 0805 | 1.000, and 0603 does not appear |
| 2.9 x 1.3 mm | SOT-23 | 1.000 |

Three details matter.

**Which outline.** A footprint carries the component body on `F.Fab`, the courtyard on
`F.CrtYd`, and the pads. The body is what a camera looking down sees; the courtyard includes
assembly clearance and overstates the part by roughly a third. The body is preferred, and the
source is recorded on every entry so an approximate one can be rejected with `body_only`.

**Size gives the body, not the pin count.** SOT-23 and SOT-143 share a body, as do LQFP-44,
LQFP-52 and LQFP-64. Size settles the family and the footprint; only appearance can count
pins. `PackagePrior.is_confident` is false whenever the runner-up is close, which is exactly
these cases, so the prior never pretends to separate what it cannot.

**Scale is the weak link, not the table.** A calibrated scale measured against a target beats
the lens model, because it absorbs the true focal length, the pixel pitch and any fixed
magnification in one number. Both assume a flat board, perpendicular and at the reported
distance, with distortion already corrected; a tilted board breaks that, and the error grows
with the tangent of the tilt. Distance error carries through proportionally, so a 2 mm error
at 150 mm is 27 micrometres on an 0805 and irrelevant, but 0.27 mm on a 20 mm connector.

Build the table with:

```powershell
python -m epr.apps.package_library build
python -m epr.apps.package_library match --size 2.0x1.25
```

## What is not here yet

The board and component detectors, and the OCR model itself. The pipeline takes regions from
whoever supplies them, which today is the simulator's ground truth, and takes markings as text
plus a confidence.

The pipeline addresses every channel with the same normalized box, which assumes the channels
are registered. That holds for the simulator and will not hold for the real head until gate
G3. Until then a hotspot can be attributed to a neighbouring component, which the plan names
as the highest risk in the project.

Training here is level 2 of the learning structure, a candidate model trained in a separate
process. Nothing in this stack may promote a candidate: that needs the validation, approval
and rollback machinery of level 3.
