"""
Running a new reading of something alongside the old one.

Every migration in this codebase that swapped where a value is read from ran
both for a while and compared them, and every one of them was first written to
log only the differences. That makes the result unreadable: the log is empty
when the two agree, and equally empty when the comparison never ran -- a
controller that stopped, a cache that never missed, a branch not reached. The
gate for switching over is "no differences", and it was being read off evidence
that could not tell those apart.

So the count is the point, not the log line. A comparison that reports
agreements is one whose silence means something.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ComparisonTally:
    """
    The running result of reading one value two ways.

    Reports at INFO on a widening cadence -- often enough at the start to show
    it is running at all, rarely enough afterwards to stay out of the way.
    """

    def __init__(self, name: str, report_at: int = 1):
        self._name = name
        self._agreed = 0
        self._differed = 0
        self._next_report_at = report_at

    @property
    def agreed(self) -> int:
        return self._agreed

    @property
    def differed(self) -> int:
        return self._differed

    def compare(self, old, new, subject: str, skip_if_none: bool = True) -> bool:
        """
        Record one comparison, logging where the two differ.

        ``new`` of None is not a difference when ``skip_if_none``: the new
        reading having nothing to say -- rows not compiled yet, a lookup with
        no answer -- is not the same as it disagreeing, and counting it as one
        would keep the gate shut for a reason unrelated to correctness.

        Returns whether the two agreed.
        """
        if new is None and skip_if_none:
            return True
        if old == new:
            self._agreed += 1
            self._report()
            return True
        self._differed += 1
        logger.info(
            f"{self._name} differs on {subject}: was {old}, reads {new} "
            f"[agreed={self._agreed} differed={self._differed}]"
        )
        return False

    def _report(self):
        total = self._agreed + self._differed
        if total < self._next_report_at:
            return
        self._next_report_at = total * 2
        logger.info(f"{self._name}: agreed={self._agreed} differed={self._differed}")


_tallies: Dict[str, ComparisonTally] = {}


def tally(name: str) -> ComparisonTally:
    """The tally for one comparison, shared by every caller that makes it.

    Keyed by name rather than held by the caller because the callers are
    functions with no instance to hang it on, and a tally per call would count
    to one and report forever.
    """
    existing: Optional[ComparisonTally] = _tallies.get(name)
    if existing is None:
        existing = ComparisonTally(name)
        _tallies[name] = existing
    return existing
