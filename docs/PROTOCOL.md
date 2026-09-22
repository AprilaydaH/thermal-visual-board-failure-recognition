# Acquisition protocol v1

Binary packet protocol for high-volume image data between the acquisition head and the PC.
Configuration and status use JSON payloads; image channels are raw binary.

All integers are little-endian, matching the STM32.

## Packet header (28 bytes)

| Offset | Size | Field | Meaning |
|---|---|---|---|
| 0 | 4 | `magic` | ASCII `EPRF` |
| 4 | 1 | `version` | Protocol version, currently 1 |
| 5 | 1 | `type` | Packet type |
| 6 | 2 | `flags` | Bit 0 `LAST_CHUNK`, others reserved |
| 8 | 8 | `frame_set_id` | Shared ID of all channels of one capture |
| 16 | 4 | `sequence` | Increments per packet for the whole connection |
| 20 | 4 | `payload_length` | Payload bytes following the header |
| 24 | 4 | `payload_crc32` | CRC-32 of the payload |

Every field is naturally aligned so the firmware can emit the header from a packed struct
without unaligned accesses.

## Packet types

| Value | Type | Payload |
|---|---|---|
| `0x01` | `METADATA` | UTF-8 JSON frame-set metadata |
| `0x02` | `RGB` | Raw RGB888 pixels |
| `0x03` | `NIR` | Raw MONO8 pixels |
| `0x04` | `THERMAL` | Raw 16-bit radiometric values, centikelvin |
| `0x05` | `FRAME_SET_END` | Empty |
| `0x10` | `STATUS` | UTF-8 JSON device status |

## Frame-set transmission

```text
METADATA -> RGB[...] -> NIR[...] -> THERMAL[...] -> FRAME_SET_END
```

The metadata packet describes every channel: width, height and pixel format. It is low volume,
so JSON is used for extensibility. Image channels carry no descriptor of their own and are
validated against the metadata.

Channels larger than the maximum payload are split into chunks. Every chunk carries the same
`type` and `frame_set_id`; only the final chunk sets `LAST_CHUNK`. Chunks are concatenated in
arrival order.

## Receiver rules

- A wrong magic or version, a payload length beyond the configured maximum, or a CRC mismatch
  makes the packet invalid.
- `sequence` must increase by exactly one. A gap counts as dropped packets and discards the
  frame set being assembled.
- A frame set is accepted only when metadata and all three channels arrived and each channel
  byte count matches its descriptor. Missing channels are never guessed.
- Raw payloads are written to storage unmodified.
