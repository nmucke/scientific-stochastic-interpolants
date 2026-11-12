"""Run ``<script> [args...]`` in-process and print the CUDA peak memory at exit.

    python peakmem_run.py run.py case=urban ...

Prints ``[peakmem] allocated=<MiB> reserved=<MiB>`` (torch caching allocator) to
stderr when the process ends, also on failure. Needed because nvidia-smi does not
report per-slice memory on MIG. Changes nothing else: argv[0] is the script, and
the script runs as ``__main__`` with its own ``__file__`` (Hydra resolves its
config_path from it).
"""
import atexit
import os
import runpy
import sys


def _report():
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            mib = 2**20
            print(f"[peakmem] allocated={torch.cuda.max_memory_allocated() / mib:.0f} "
                  f"reserved={torch.cuda.max_memory_reserved() / mib:.0f}", file=sys.stderr, flush=True)
    except Exception as exc:  # pragma: no cover
        print(f"[peakmem] unavailable: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    atexit.register(_report)
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path[0] = os.path.dirname(os.path.abspath(script))  # as `python <script>`
    runpy.run_path(script, run_name="__main__")
