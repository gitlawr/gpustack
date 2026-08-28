"""
When to restart a workload that stopped, and when to give up on it.

The policy is the same shape everywhere -- exponential delay, capped, with a
stability window that forgives an old crash -- but the parameters are not, and
neither is where the attempt count is kept. So this holds the arithmetic and
the decision; callers supply the count and the timestamp from wherever they
store them.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class RestartActionEnum(str, Enum):
    RESTART = "restart"
    WAIT = "wait"
    GIVE_UP = "give_up"
    """The budget is spent. The caller parks the workload rather than
    restarting it forever with nothing to show for it."""


@dataclass(frozen=True)
class RestartDecision:
    action: RestartActionEnum
    attempt: int = 0
    """1-based number of the restart being authorised, when action is
    RESTART."""
    delay_remaining_seconds: float = 0.0
    """How much of the backoff window is left, when action is WAIT."""


@dataclass(frozen=True)
class RestartBudget:
    """
    Args:
        base_delay_seconds: Delay of the first backed-off restart.
        max_delay_seconds: Ceiling the doubling stops at.
        max_attempts: Consecutive restarts allowed before giving up. None
            keeps retrying, which is what a workload that may recover from an
            outside failure wants.
        reset_after_seconds: How long a workload must stay up, measured from
            its last restart, before the consecutive count is forgiven. None
            leaves the count to the caller.
        first_attempt_immediate: Restart the first crash with no delay. Cheap
            recovery from a one-off, at the cost of one tight loop before the
            backoff engages.
    """

    base_delay_seconds: float
    max_delay_seconds: float
    max_attempts: Optional[int] = None
    reset_after_seconds: Optional[float] = None
    first_attempt_immediate: bool = False

    def delay_for(self, attempts: int) -> float:
        """The delay owed before the restart following ``attempts`` earlier ones."""
        if attempts <= 0:
            return 0.0 if self.first_attempt_immediate else self.base_delay_seconds
        exponent = attempts - 1 if self.first_attempt_immediate else attempts
        return min(self.base_delay_seconds * 2**exponent, self.max_delay_seconds)

    def decide(
        self,
        attempts: int,
        last_attempt_at: Optional[datetime],
        now: datetime,
    ) -> RestartDecision:
        """
        Args:
            attempts: Consecutive restarts already made.
            last_attempt_at: When the last one was made. None means the
                workload has not been restarted yet, so nothing is owed.
            now: Current time, passed in so the decision stays testable.
        """
        attempts = max(attempts or 0, 0)

        if self.max_attempts is not None and attempts >= self.max_attempts:
            return RestartDecision(RestartActionEnum.GIVE_UP, attempt=attempts)

        delay = self.delay_for(attempts)
        if delay and last_attempt_at is not None:
            elapsed = (now - last_attempt_at).total_seconds()
            if elapsed < delay:
                return RestartDecision(
                    RestartActionEnum.WAIT,
                    attempt=attempts,
                    delay_remaining_seconds=delay - elapsed,
                )

        return RestartDecision(RestartActionEnum.RESTART, attempt=attempts + 1)

    def should_forgive(
        self,
        attempts: int,
        last_attempt_at: Optional[datetime],
        now: datetime,
    ) -> bool:
        """
        Whether a workload that is up again has been up long enough to get its
        budget back. A workload that breaks out of a crash loop and fails again
        much later should meet a fresh set of attempts, not the tail of the old
        ones.
        """
        if self.reset_after_seconds is None:
            return False
        if attempts <= 0 or last_attempt_at is None:
            return False
        return (now - last_attempt_at).total_seconds() >= self.reset_after_seconds
