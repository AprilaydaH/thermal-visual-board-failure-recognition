# Acquisition head (STM32)

Development platform: **STM32N6570 DK**.

Firmware owns initialization, RGB / NIR / thermal capture, timing, USB bulk transport, diagnostics. It does not run the production recognizer.

First goal: stream one synchronized RGB + thermal frame set to the PC.
