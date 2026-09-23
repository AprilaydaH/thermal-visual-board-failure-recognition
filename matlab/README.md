# MATLAB access to stored frame sets

G1 data is ordinary files. MATLAB does not need the Python package.

```text
<project>/raw/
  rgb/000000000004.png      RGB
  nir/000000000004.png      850 nm
  thermal/000000000004.tiff radiometric uint16, centikelvin
  metadata/000000000004.json
```

Celsius is `double(thermal) / 100 - 273.15`.

From the repository root, after `epr inspect --simulate` (or `epr acquire`):

```matlab
addpath('matlab')
inspect_frameset('data/projects/g1-smoke')           % click on RGB
inspect_frameset('data/projects/g1-smoke', 0.45, 0.42)  % hot QFP
inspect_frameset('data/projects/g1-smoke', 0.08, 0.08)  % board corner
```

`0.45, 0.42` should read about 44 C and `0.08, 0.08` about 22 C on the simulated set.

This tests storage and the G1 mapping. It does not train or run the CNN/LNN; those stay in Python (export to ONNX if you later want MATLAB inference).
