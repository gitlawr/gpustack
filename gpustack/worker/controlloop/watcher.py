import asyncio
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_RETRY_INTERVAL_SECONDS = 5


async def watch_forever(
    name: str,
    awatch: Callable[..., Any],
    callback: Optional[Callable] = None,
    retry_interval: float = DEFAULT_RETRY_INTERVAL_SECONDS,
):
    """
    Consume a resource's watch stream, reconnecting until cancelled.

    A dropped stream is a routine event (server restart, proxy timeout), so it
    is retried rather than propagated; cancellation is the only way out, since
    an unhandled exception here would silently stop the worker reacting to a
    whole resource kind.

    Args:
        name: The resource kind, for log messages.
        awatch: The clientset's ``awatch`` for that kind.
        callback: Invoked per event. None keeps the client's cache warm
            without reacting to individual events.
        retry_interval: Seconds to wait before reconnecting after a failure.
    """
    logger.debug(f"Watching {name}.")
    while True:
        try:
            await awatch(callback=callback)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error watching {name}: {e}")
            await asyncio.sleep(retry_interval)
