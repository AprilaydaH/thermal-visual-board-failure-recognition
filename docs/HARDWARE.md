# V1 hardware interface

The acquisition head is a **sensor gadget**, not a recognizer. It captures a
[frame set](FRAME_SET.md), stamps one `frame_set_id`, and streams it over USB. The PC owns
detection, OCR, fusion and learning.

This document is the hardware contract the firmware is written against. Parts marked
**selected** are decided. Parts marked **open** are still a purchase choice, but the
electrical role is not.

## Block diagram

```text
                    850 nm LED ring ── illumination GPIO / current sense
                    white LED ring  ── illumination GPIO

  PCB under test
        │
        ├─ RGB macro ──── CSI-2 ────┐
        ├─ NIR (or RGB+filter) ─────┤
        ├─ FLIR Lepton 3.5 ─ VoSPI ─┤     STM32N6570 DK
        ├─ ToF distance ──── I2C ───┤           │
        └─ ambient T/RH ──── I2C ───┘           │
                                                USB 2.0 HS bulk
                                                │
                                                PC acquisition service
                                                │
                                                FrameSet storage (raw, immutable)
```

Trigger is **internal**: the timing manager on the STM32 asserts capture on all imagers for
one `frame_set_id`. There is no external encoder. A technician click is not a hardware
trigger; it is a PC-side lookup into an already stored set.

## Selected parts

| Role | Choice | Interface to MCU | Notes |
|---|---|---|---|
| MCU / first head | STM32N6570 Discovery Kit | — | NPU stays unused in V1; the kit is a camera + USB host, not an edge recognizer |
| Thermal | FLIR Lepton 3.5 radiometric | VoSPI (SPI) + CCI (I²C-like), VSYNC GPIO | 160×120, T-linear / radiometric, export frame rate &lt; 9 Hz. **Do not** colourize on the MCU. |
| USB | Kit HS USB device | Bulk, protocol v1 | JSON only for metadata and status; image payloads are raw |
| Stand | Rigid adjustable macro stand | — | Repeatable board pose; highest risk in the plan is a hotspot on the wrong part |

Lepton is not a CSI-2 camera. It cannot plug into the DK camera connector. Prototype options:

1. **Preferred for G1:** Lepton on VoSPI/CCI bit-banged or DMA-SPI from the STM32, firmware owns the packet.
2. **Lab shortcut:** PureThermal / equivalent UVC bridge into the PC, STM32 still owns RGB/NIR and the `frame_set_id`. Only acceptable until G1; the production path is (1) so thermal timestamps live on the same clock.

## Open purchases (role is fixed)

| Role | Requirement | Interface | Open choice |
|---|---|---|---|
| RGB | ≥ 5 MP, macro, global or short rolling shutter, C/CS or M12 | MIPI CSI-2 on the DK camera connector | Exact module (kit camera vs industrial 5 MP) after G2 optics tests |
| NIR | 850 nm band; either a second mono camera or the RGB sensor with a switched 850 nm illuminator and IR-pass filter | CSI-2 or the RGB path plus illumination GPIO | Dual-camera vs filter-wheel / LED-only on one sensor |
| Illumination | White for RGB/OCR; 850 nm for NIR; no mixed lighting in one frame set | GPIO + current sense into `led_current_ma` | LED ring vendor; must not saturate Lepton |
| Distance | Millimetre-class ToF, 50–300 mm working range | I²C | VL53L1X-class or equivalent; feeds `geometry.distance_mm` |
| Ambient | Temperature (required), humidity (required) | I²C | SHT3x / SHT4x class |
| Power | 5 V from USB for the kit; separate 5 V barrel if LEDs exceed USB budget | Dedicated LED supply, common ground | Measure LED current before locking the PSU |
| Board fixture | Repeatable XY and height, no 30° handheld tilt | Mechanical | Custom plate after the first five sample PCBs exist |

## Synchronization

| Event | Owner | Effect on the frame set |
|---|---|---|
| Capture pulse | STM32 timing manager | Same `frame_set_id` on RGB, NIR, thermal, distance, ambient |
| Timestamp | STM32 monotonic clock | `timestamp_ns` |
| Lepton FFC / shutter | Thermal controller | `thermal.shutter_state`; a closed shutter **rejects** the set |
| Illumination mode | GPIO before exposure | `environment.illumination_state`; RGB and NIR sets are separate if lighting differs |
| Dropped SPI/USB | Diagnostics | `sensor_status.* = fault` and the PC discards the set |

RGB and thermal cannot be optically coincident: different FOV and a baseline of centimetres. Registration is a **PC** problem (gate G3) using `distance_mm` and `calibration_id`. The head must still freeze pose between the three exposures of one set; a moving board invalidates the ID.

## Power and connectors (prototype)

- STM32N6570 DK powered from USB while developing protocol.
- Lepton: 2.8 V / 1.2 V as in the module datasheet, never 5 V on the core.
- LED rings: separate 5 V, PWM or current drivers, sense resistor to an ADC for `led_current_ma`.
- One shielded USB-C or USB-B to the PC; no Ethernet in V1.
- Break out Lepton SPI, CCI, VSYNC, RESET on a 2.54 mm header so the module can be swapped.

## What firmware must implement first

Matches [IMMEDIATE_ACTIONS.md](IMMEDIATE_ACTIONS.md) gates G1–G2:

1. USB bulk with [PROTOCOL.md](PROTOCOL.md) v1.
2. One RGB frame + metadata, stored raw on the PC.
3. One radiometric Lepton frame + metadata, stored raw.
4. One combined frame set (RGB + thermal) with a shared `frame_set_id`.
5. Distance and ambient in the same metadata packet, even if dummy-stable, so the PC contract does not change later.

NIR can follow RGB once illumination exists. Recognition models are not a firmware deliverable.

## Out of scope on the head

NPU inference, colourized thermal video, onboard OCR, Wi-Fi, battery, handheld trigger. Those belong to later product gates, not V1.
