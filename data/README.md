# Dataset convention

Split train / validation / test by **physical PCB**, not by image.

```
data/
  raw/                 # immutable captures (not committed)
  projects/            # inspection sessions
  labels/              # reviewed annotations
  models/              # versioned packages (not large weights in git)
```

Record who labelled each example and when it changed. Include blur, glare, dirt and unknown parts. Keep raw images linked to crops through frame-set IDs.
