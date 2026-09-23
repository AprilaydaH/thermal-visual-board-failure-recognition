# Multimodal Electronic Parts Recognition System

**Thermal and visual electronic boards and parts analysis and failure recognition**

Author: Dereje Hailemariam  
Plan: v1.0 · 22 September 2026 · planning baseline (VDI 2220 / VDI 2221)

A bench-oriented inspection aid for repair and engineering labs. An **STM32 acquisition head** captures synchronized **RGB**, **850 nm NIR** and **radiometric thermal** frames. A **Windows PC** application detects components, reads markings, assigns temperatures and learns from technician corrections.

## V1 in scope

- Stationary / stand-mounted inspection of unpowered and powered PCBs
- High-resolution RGB macro capture (detection + OCR)
- Controlled 850 nm NIR capture
- Radiometric thermal capture (hotspot and heating rate)
- Distance sensing, illumination, repeatable registration
- Local Windows app: acquisition, annotation, recognition, supervised learning, reports
- Confirmation / correction with model versioning and rollback

## V1 out of scope

Exact ID of every unmarked part · electrical values that are not optically encoded · autonomous diagnosis · SWIR / hyperspectral / Raman / XRF / X-ray · medical temperature claims · cloud-mandatory processing · handheld battery product before the bench workflow is validated

## Architecture

| Layer | Role |
|---|---|
| STM32N6570 DK head | Deterministic sensor control and USB streaming |
| Windows PC | Processing, recognition, learning, storage |
| Sensors (prototype) | 5 MP macro RGB, 850 nm NIR, FLIR Lepton 3.5, ToF distance |

The basic object is a **frame set**: RGB + NIR + raw thermal + timestamp, distance, exposure, illumination, focus, ambient data, calibration IDs. Incomplete sets are rejected.

PC stack (prototype): Python / PySide6, OpenCV, PyTorch detector, OCR, CfC/LNN for temporal fusion, ONNX Runtime, SQLite + project folders.

## First milestones

1. **Technical:** head sends one synchronized RGB + thermal frame set; PC stores **raw** data; click a visible point and see the corresponding thermal region.
2. **Product:** an external technician inspects an unfamiliar board, finds a heated part faster than today, corrects the label and reopens the saved inspection.

Recognition starts only after the data path is reliable.

## Repository layout

| Path | Content |
|---|---|
| `docs/` | Project plan and specifications |
| `firmware/` | STM32 acquisition head |
| `pc/` | Windows application and acquisition service |
| `data/` | Dataset convention (raw captures stay local, not in git) |

## Documents

- [Project plan (Word)](docs/Multimodal_Electronic_Parts_Recognition_Project_Plan.docx)
- [Software structure](docs/ARCHITECTURE.md)
- [Recognition stack](docs/RECOGNITION.md)
- [Acquisition protocol](docs/PROTOCOL.md)
- [Frame set specification](docs/FRAME_SET.md)
- [V1 hardware interface](docs/HARDWARE.md)
- [Immediate actions](docs/IMMEDIATE_ACTIONS.md)

Independent of the PickPlace / Factory I/O Abschlussprojekt.

## Install

Python 3.11 or newer. Clone the repository, then from its root:

```powershell
.\install.ps1
.\.venv\Scripts\Activate.ps1
epr acquire
```

That installs the acquisition stack (sensors, storage, simulator). Recognition needs PyTorch:

```powershell
.\install.ps1 -Ml      # any computer, CPU or the default GPU wheels
.\install.ps1 -Cuda    # this project's CUDA 12.6 pin (NVIDIA GPU)
```

On Linux or macOS: `./install.sh` or `./install.sh --ml`.

Manual equivalent:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

| Extra | What it adds |
|---|---|
| (none) | Acquisition, storage, tests, `epr acquire` / `epr packages` |
| `[ml]` | CNN, LNN, detector training (`epr train`, `epr demo`) |
| `pc/requirements-ml.txt` | CUDA 12.6 PyTorch, only if `[ml]` is not enough |

`epr` is the installed command. Data, models and projects go in this checkout's `data/` folder, or in `%LOCALAPPDATA%\epr` on a machine without the source tree. Override with `EPR_HOME`.

Firmware for the STM32 head is not a Python package; it stays in `firmware/` and is built with STM32Cube.
