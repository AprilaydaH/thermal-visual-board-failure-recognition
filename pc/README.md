# Windows PC application

Prototype stack: Python + PySide6, OpenCV, PyTorch detector, OCR, CfC/LNN temporal fusion, ONNX Runtime, SQLite.

The PC owns image processing, recognition, learning and storage. Raw frame sets are never overwritten.

First goal: connect to the head, store raw RGB + thermal data, click a visible point and inspect the mapped thermal region.

## Environment

Python 3.11, virtual environment at the repository root.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r pc\requirements-dev.txt
```

| File | Content |
|---|---|
| `requirements.txt` | Acquisition, storage and UI stack (milestone 1) |
| `requirements-dev.txt` | The above plus pytest, pytest-qt, ruff |
| `requirements-ml.txt` | Recognition stack, installed separately from G4 onwards |

`requirements-ml.txt` is pinned to the CUDA 12.6 build of PyTorch; the header explains how to
switch to CPU wheels. The acquisition stack does not import PyTorch, so a machine that only
records data can skip it. The model tests skip themselves when it is absent.

Install the application package itself in editable mode:

```powershell
pip install -e pc --no-deps
```

Checks, run from `pc/`:

```powershell
ruff check .
pytest
```

## Source layout

Implements the decomposition of [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) §5 under a single
`epr` namespace package, so that module names such as `core` or `device` do not occupy global
import names.

```text
pc/src/epr/
├── core/
│   ├── domain_models/   frame set, channels, sensor status
│   └── errors.py
├── device/
│   ├── protocol/        packet wire format and frame-set framing
│   ├── simulator/       simulated acquisition head
│   ├── acquisition.py   packet validation and frame-set reassembly
│   └── transport.py     byte-stream adapters, USB and Ethernet later
├── processing/
│   ├── thermal/         radiometric conversion and thermal features
│   └── crops.py         component cropping
├── recognition/
│   ├── classification/  dual-stream RGB and NIR CNN
│   ├── lnn/             closed-form continuous-time cells
│   ├── fusion/          feature layout and the LNN fusion model
│   ├── ocr/             marking features
│   ├── unknown_detection/
│   ├── pipeline.py      frame sets and regions to predictions
│   └── export.py        ONNX export
├── learning/
│   └── training/        candidate model training
├── storage/
│   └── file_storage/    immutable raw recording
└── apps/
    ├── acquisition_service/
    └── recognition_demo/
```

## Acquisition service

Runs the full data path against the simulated gadget and writes raw frame sets to a project
directory. No hardware required.

```powershell
python -m epr.apps.acquisition_service --project data/projects/demo --frames 5
```

`--frame-interval` sets the simulated seconds between frame sets and therefore how far the
components have heated up.

## Recognition demo

Runs a simulated heating sequence through the CNN and the LNN and prints what the interface
would show. See [docs/RECOGNITION.md](../docs/RECOGNITION.md) for the models themselves.

```powershell
python -m epr.apps.recognition_demo --frames 6 --device auto
```

The models are untrained, so the classes are noise and the unknown detector rejects
everything, which is the correct behaviour. The thermal columns are real measurements.

## Training service

Pretrains the RGB stream of the component CNN on the public WACV 2019 PCB dataset. Downloads
about 284 MB on first run into `data/external/`, which is not committed.

```powershell
python -m epr.apps.training_service --download --epochs 60 --device cuda
```

This produces a candidate checkpoint, never a production model. Only the RGB stream is
trained; public data has no 850 nm or thermal channel, so it cannot train the NIR stream or
the LNN. See [docs/RECOGNITION.md](../docs/RECOGNITION.md).

The detector is trained by the same service on the same boxes, class-agnostic:

```powershell
python -m epr.apps.training_service detector --epochs 60 --device cuda
```

## Package library

Reads the installed KiCad footprint libraries into a table of physical package sizes, then
answers what a measured component could be. Not trained, and offline.

```powershell
python -m epr.apps.package_library build
python -m epr.apps.package_library match --size 2.0x1.25
python -m epr.apps.package_library scale --distance 150 --pixels 140
```

The table is written to `data/external/package_library.json`, which is not committed: it is
derived from a local KiCad install and rebuilt with one command.
