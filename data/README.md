# Dataset convention

Split train / validation / test by **physical PCB**, not by image.

```
data/
  raw/                 # immutable captures (not committed)
  projects/            # inspection sessions
  labels/              # reviewed annotations
  external/            # downloaded public datasets and crop caches (not committed)
  models/              # versioned packages (not large weights in git)
```

Record who labelled each example and when it changed. Include blur, glare, dirt and unknown parts. Keep raw images linked to crops through frame-set IDs.

## External datasets

None of these can train the LNN: none carries NIR or radiometric thermal over time. They are
visual-only, and each has a domain gap against a macro rig looking at a powered board.

| Directory | Content | Used by |
|---|---|---|
| `pcb_wacv_2019/` | 47 populated board scans, PASCAL VOC boxes, 31 component types | RGB stream pretraining, component detector |
| `cache/` | Crop caches built from the above, with a fingerprint per build | `epr.apps.training_service cnn` |
| `electrocom61/` | 2071 photographs of loose components, 12667 boxes, YOLO, 61 classes, CC BY 4.0 | nothing yet |
| `pcb_defects/` | 10668 bare-board images, YOLO, 6 copper and drilling defects | nothing yet |
| `scraped_components/` | 3531 files in 20 class folders, of which 3194 decode as images, unlabelled | nothing yet |

### pcb_wacv_2019

Kuo et al., WACV 2019. Downloaded by `epr.apps.training_service cnn --download`. Board scans
are named `ACM-109_Top` and `ACM-109_Bottom`: two sides of one PCB, so the loader merges them
before splitting, turning 47 scans into 34 physical boards.

### electrocom61

Roboflow export of ElectroCom-60 v5, CC BY 4.0. Loose components photographed on a table with
a phone camera, so the geometry is unlike a board-mounted part. Its `data.yaml` shipped with
`../train/images` paths that did not resolve from its own location; corrected to be relative
to the file.

Well formed otherwise: 1454 / 412 / 205 images per split, every image paired with a label, no
empty label files, all 640x640. All 61 declared classes are annotated and none has fewer than
20 instances, so the taxonomy can be used as published rather than pruned like the WACV one.

### pcb_defects

Bare, unpopulated boards annotated for manufacturing faults. This is a different inspection
stage from the V1 scope of powered populated boards, and would need a model that the
architecture does not currently define.

Two faults to handle before training on it:

- **Label names disagree with image names.** Every image is `..._600.jpg`, but 2164 train,
  264 val and 239 test labels are named `..._256.txt`. The coordinates are correct and
  normalized; only the name differs. A loader that derives the label path from the image name
  finds nothing for those 2667 images and feeds them in as defect-free backgrounds, teaching
  the model to suppress the defects they actually contain. Match on the stem with the trailing
  size token removed.
- **The splits are not independent.** All ten board templates appear in train, val and test,
  differing only by lighting and rotation. Re-split by template before believing any score.

### scraped_components

A web scrape, not a released dataset: mixed and broken extensions including `.php`, `.view`
and `.image`, no annotations, no licence and no provenance. Treat as unverified. Each class
folder originally nested a second folder of the same name, which has been flattened.

337 of the 3531 files do not decode as images at all. The remaining 3194 span 1135 distinct
resolutions from 19x19 to 7682x5833, so anything built on this needs a size floor and a
decode check before the images reach a loader.
