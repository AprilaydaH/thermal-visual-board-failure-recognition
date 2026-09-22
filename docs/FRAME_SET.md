# Frame set specification

The PC rejects incomplete or mismatched frame sets. Do not guess missing channels.

## Identity

| Field | Meaning |
|---|---|
| device_id | Acquisition head serial |
| session_id | Inspection session |
| frame_set_id | Common ID for all channels of one capture |
| timestamp | Monotonic capture time |

## Channels

| Group | Fields |
|---|---|
| RGB | Image, exposure, gain, focus, white balance |
| NIR | Image, wavelength (850 nm), LED current, exposure, gain |
| Thermal | Raw values, calibration mode, shutter status, sensor temperature |
| Geometry | Distance, camera pose if available, calibration profile ID |
| Environment | Ambient temperature, humidity, illumination status |

## Storage rules

- Keep **raw** source data. Never overwrite it with processed images.
- Link crops and labels to immutable frame-set and region IDs.
- Calibration profiles are keyed by device serial and date.
- Packet protocol (USB bulk): type, frame_set_id, payload length, integrity check.
