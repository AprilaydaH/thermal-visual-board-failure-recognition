# Immediate actions (plan §16)

## First 30 days

1. One-page user interview guide; contact at least ten repair technicians.
2. Acquire STM32N6570 DK, Lepton 3.5 interface, macro lens, illumination parts.
3. Select five representative PCBs (dense smartphone, laptop, industrial).
4. Rigid adjustable camera stand with repeatable board positioning.
5. This repository, dataset folder convention, frame-set specification.
6. Stream and save the first RGB image with complete metadata.
7. Stream and save the first radiometric thermal frame.
8. Document bandwidth, dropped frames, temperatures and image quality.

## Gates (short)

| Gate | Criterion |
|---|---|
| G0 Problem | Users confirm a valuable task and price range |
| G1 Acquisition | Stable RGB and thermal data reach the PC; click RGB → thermal temperature (`epr inspect`) |
| G2 Optics | Markings and heated parts observable at target distance |
| G3 Calibration | RGB, NIR and thermal repeatably registered |
| G4 Recognition | Baseline metrics on unseen boards |
| G5 Pilot | Technicians show measurable usefulness |
| G6 Product | Cost, safety, support and demand justify custom hardware |

Highest risk: assigning a thermal hotspot to the **wrong** visible component. Stand, distance sensing, calibration and registration tests have the same priority as the model.
