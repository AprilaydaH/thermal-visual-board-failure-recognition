# Frame set specification

The frame set is the **central contract** of the system. Firmware, USB packets, storage and
recognition all speak this object. The PC rejects incomplete or mismatched sets. Missing
channels are never guessed.

Protocol version 1 is defined in [PROTOCOL.md](PROTOCOL.md). The Pydantic model is
`epr.core.domain_models.frame_set.FrameSetMetadata`. Extra fields are forbidden, so a
firmware change that adds a key is a protocol bump, not a silent extension.

## Object

```text
FrameSet
├── device_id
├── session_id
├── frame_set_id
├── timestamp_ns
├── rgb
│   ├── image (width, height, rgb888)
│   ├── exposure_us
│   ├── gain_db
│   ├── focus_position
│   └── white_balance_k
├── nir
│   ├── image (width, height, mono8)
│   ├── wavelength_nm          # 850
│   ├── led_current_ma
│   ├── exposure_us
│   └── gain_db
├── thermal
│   ├── image (width, height, raw16 centikelvin)
│   ├── calibration_mode
│   ├── shutter_state
│   └── sensor_temperature_c
├── geometry
│   ├── distance_mm
│   ├── calibration_profile_id
│   └── camera_pose            # optional
├── environment
│   ├── ambient_temperature_c
│   ├── humidity_percent
│   └── illumination_state     # off | white | nir_850
├── calibration_id
└── sensor_status              # rgb, nir, thermal, distance, environment
                               # each ok | degraded | fault
```

Raw thermal values are transmitted, never a colourized JPEG. Recognition may derive Celsius
later; storage keeps the centikelvin array.

## Why this shape

A later module can ignore a block it does not use, but it cannot reconstruct one that never
arrived. Registration needs `calibration_id` and `distance_mm`. The LNN needs `timestamp_ns`
so `dt` is real elapsed time, not a frame index. Unknown-sensor handling needs `sensor_status`
so a dead NIR LED degrades one feature block instead of the whole set.

## Identity

| Field | Meaning |
|---|---|
| device_id | Acquisition head serial |
| session_id | Inspection session |
| frame_set_id | Common ID for all channels of one capture |
| timestamp_ns | Monotonic capture time on the head clock |

## Storage rules

- Keep **raw** source data. Never overwrite it with processed images.
- Link crops and labels to immutable frame-set and region IDs.
- Calibration profiles are keyed by device serial and date.
- Packet protocol (USB bulk): type, frame_set_id, payload length, integrity check.
