"""The installed `epr` command dispatches without importing PyTorch."""

import subprocess
import sys

from epr.apps.cli import COMMANDS, main


def test_help_lists_the_services():
    assert main(["--help"]) == 0
    assert main([]) == 0


def test_unknown_command_is_rejected():
    assert main(["not-a-command"]) == 2


def test_every_service_has_a_main():
    import importlib

    for module_name in COMMANDS.values():
        if "training" in module_name or "recognition_demo" in module_name:
            continue
        module = importlib.import_module(module_name)
        assert callable(module.main)


def test_cli_import_does_not_load_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import epr.apps.cli, sys; print('torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"
