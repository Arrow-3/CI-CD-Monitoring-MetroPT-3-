from __future__ import annotations
import atexit
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
import sys

# Startup order: consumers before producers. DataGenerator is last.
SERVICES: list[tuple[str, list[str]]] = [
    ("gate",  [sys.executable, "-m", "metropt.services.data_gate"]),
    ("fe",    [sys.executable, "-m", "metropt.services.feature_extractor"]),
    ("model", [sys.executable, "-m", "metropt.services.primary_model"]),
    ("dr",    [sys.executable, "-m", "metropt.services.dim_reducer"]),
    ("dm",    [sys.executable, "-m", "metropt.services.distribution_monitor"]),
    ("tl",    [sys.executable, "-m", "metropt.services.transfer_learning"]),
    ("gen",   [sys.executable, "-m", "metropt.services.data_generator"]),
]

# ANSI color codes per service — makes the interleaved log readable.
COLORS = ["\033[36m", "\033[32m", "\033[33m", "\033[35m",
          "\033[34m", "\033[31m", "\033[37m"]
RESET = "\033[0m"


@dataclass
class Managed:
    name: str
    color: str
    proc: subprocess.Popen


processes: list[Managed] = []


def _pump(mp: Managed) -> None:
    """Read the subprocess's stdout line by line, prefix, print."""
    prefix = f"{mp.color}[{mp.name:<5}]{RESET}"
    for raw in mp.proc.stdout:
        line = raw.rstrip()
        print(f"{prefix} {line}", flush=True)


def _shutdown(*_):
    print("\n\033[1m[orchestrator]\033[0m shutting down...", flush=True)
    # Reverse order: kill producer first, then consumers.
    for mp in reversed(processes):
        if mp.proc.poll() is None:
            mp.proc.terminate()
    for mp in reversed(processes):
        try:
            mp.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"[orchestrator] killing {mp.name}", flush=True)
            mp.proc.kill()
    print("[orchestrator] all services stopped.", flush=True)


def main() -> None:
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    atexit.register(_shutdown)

    print("\033[1m[orchestrator]\033[0m starting all services...", flush=True)
    for i, (name, cmd) in enumerate(SERVICES):
        color = COLORS[i % len(COLORS)]
        print(f"{color}[{name:<5}]{RESET} starting: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        mp = Managed(name=name, color=color, proc=proc)
        processes.append(mp)
        threading.Thread(target=_pump, args=(mp,), daemon=True).start()
        # Small gap so downstream consumers subscribe before upstream produces.
        time.sleep(1.5 if name != "gen" else 0)

    print("\033[1m[orchestrator]\033[0m all services started. Ctrl-C to stop.\n",
          flush=True)

    # Wait for any subprocess to die (usually means someone crashed).
    while True:
        time.sleep(1)
        for mp in processes:
            if mp.proc.poll() is not None:
                print(f"\033[1m[orchestrator]\033[0m {mp.name} exited "
                      f"with code {mp.proc.returncode}", flush=True)
                _shutdown()
                sys.exit(mp.proc.returncode)


if __name__ == "__main__":
    main()