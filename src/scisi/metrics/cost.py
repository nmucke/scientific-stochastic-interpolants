"""Cost metrics (spec Section 3d): NFE counting and wall-clock timing.

Dependency-light helpers the samplers can wire in later:

* :class:`NFECounter` -- a counter object the samplers increment on every
  network evaluation (including guidance-gradient evals). Usable as a context
  manager that resets on entry.
* :class:`StepTimer` -- a ``perf_counter``-based wall-clock timer for the
  seconds-per-assimilation-step metric. Usable as a context manager.
"""

import time
from types import TracebackType
from typing import Optional, Type

__all__ = ["NFECounter", "StepTimer"]


class NFECounter:
    """Counts network function evaluations (NFE) per assimilation step.

    A sampler holds one of these and calls :meth:`increment` on every network
    evaluation (forward velocity/score eval *and* any guidance-gradient eval).
    The counter can be reused across steps via :meth:`reset`, or used as a
    context manager that resets on entry:

        nfe = NFECounter()
        with nfe:
            ...  # nfe.increment() inside the sampler loop
        steps = nfe.count
    """

    def __init__(self) -> None:
        self.count: int = 0

    def increment(self, n: int = 1) -> None:
        """Add ``n`` network evaluations to the running count."""
        self.count += n

    def reset(self) -> None:
        """Reset the count to zero."""
        self.count = 0

    def __enter__(self) -> "NFECounter":
        self.reset()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        return None


class StepTimer:
    """Wall-clock timer (``time.perf_counter``) for one assimilation step.

    Use as a context manager; the elapsed seconds are available afterwards on
    :attr:`elapsed`:

        timer = StepTimer()
        with timer:
            ...  # one assimilation step
        seconds = timer.elapsed
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: Optional[float] = None

    def start(self) -> None:
        """Begin timing."""
        self._start = time.perf_counter()

    def stop(self) -> float:
        """Stop timing, store and return the elapsed seconds."""
        if self._start is None:
            raise RuntimeError("StepTimer.stop() called before start().")
        self.elapsed = time.perf_counter() - self._start
        self._start = None
        return self.elapsed

    def __enter__(self) -> "StepTimer":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.stop()
