# Acquisition head (STM32)

Development platform: **STM32N6570 DK**. Sensors, power and USB are specified in
[docs/HARDWARE.md](../docs/HARDWARE.md). The on-wire contract is
[docs/PROTOCOL.md](../docs/PROTOCOL.md). The object they both implement is
[docs/FRAME_SET.md](../docs/FRAME_SET.md).

Firmware owns initialization, RGB / NIR / thermal capture, timing, USB bulk transport and
diagnostics. It does not run the production recognizer.

First goal: stream one synchronized RGB + thermal frame set to the PC.
