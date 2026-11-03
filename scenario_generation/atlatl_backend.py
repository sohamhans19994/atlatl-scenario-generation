"""Integration with a separately installed, authorized AtlAtl checkout."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class AtlatlNotInstalledError(RuntimeError):
    """Raised when the separately distributed AtlAtl package is unavailable."""


@dataclass(frozen=True)
class ReplayPaths:
    blue: Path
    red: Path


def find_atlatl_root() -> Path:
    """Return the installation root containing the external ``server`` package."""
    try:
        server_package = importlib.import_module("server")
        importlib.import_module("server.server")
    except ModuleNotFoundError as exc:
        raise AtlatlNotInstalledError(
            "AtlAtl is not installed in this Python environment. Obtain access "
            "from the AtlAtl author, then install that checkout before running "
            "scenario evaluation."
        ) from exc

    package_file = getattr(server_package, "__file__", None)
    if package_file:
        return Path(package_file).resolve().parent.parent

    package_paths = list(getattr(server_package, "__path__", []))
    if package_paths:
        return Path(package_paths[0]).resolve().parent

    raise AtlatlNotInstalledError(
        "The installed 'server' package could not be resolved to an AtlAtl root."
    )


class AtlatlBackend:
    """Run scenarios using the real AtlAtl command-line server."""

    def __init__(self, timeout: float = 10_000):
        self.timeout = timeout
        self.atlatl_root = find_atlatl_root()

    def run(self, scenario: dict[str, Any], output_dir: Path) -> ReplayPaths:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        scenario_path = output_dir / "scenario.scn"
        blue_replay = output_dir / "bluereplay.js"
        red_replay = output_dir / "redreplay.js"
        scenario_path.write_text(json.dumps(scenario, indent=2), encoding="utf-8")

        for replay_path in (blue_replay, red_replay):
            replay_path.unlink(missing_ok=True)

        argv = [
            sys.executable,
            "-m",
            "server.server",
            str(scenario_path),
            "--redAI",
            "pass-agg",
            "--blueAI",
            "pass-agg",
            "--blueReplay",
            str(blue_replay),
            "--redReplay",
            str(red_replay),
            "--exitWhenTerminal",
            "--nReps",
            "1",
        ]

        try:
            subprocess.run(
                argv,
                check=True,
                cwd=self.atlatl_root,
                env=os.environ.copy(),
                timeout=self.timeout,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"AtlAtl did not finish within {self.timeout} seconds."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"AtlAtl exited with status {exc.returncode}."
            ) from exc

        missing = [path.name for path in (blue_replay, red_replay) if not path.exists()]
        if missing:
            raise RuntimeError(
                "AtlAtl completed without producing expected replay files: "
                + ", ".join(missing)
            )

        return ReplayPaths(blue=blue_replay, red=red_replay)
