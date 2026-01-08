import asyncio
import logging
import threading

logger = logging.getLogger(__name__)

# Global set to hold background tasks, preventing them from being garbage collected
_pending_background_tasks = set()
# Track the main event loop for thread-safe task creation
_main_event_loop = None
_loop_lock = threading.Lock()


def set_main_event_loop(loop):
    """Set the main event loop for thread-safe task creation."""
    global _main_event_loop
    with _loop_lock:
        _main_event_loop = loop
        logger.debug(f"Set main event loop: {loop}")


def create_background_task(coro, *, name: str = None):  # noqa: C901
    """
    Create an asyncio task that is protected from garbage collection.

    This is a wrapper around asyncio.create_task() that maintains a strong reference
    to the task to prevent it from being garbage collected before it runs.
    It works correctly even when called from a different thread than the event loop.

    Args:
        coro: The coroutine to run as a task
        name: Optional name for the task

    Returns:
        The created Task object or Future

    Example:
        task = create_background_task(
            event_bus.publish("topic", event),
            name="publish_modelinstance_deleted"
        )
    """
    global _main_event_loop

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, try to get or set the main loop
        with _loop_lock:
            if _main_event_loop is None:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        _main_event_loop = loop
                    else:
                        logger.warning(
                            f"Found event loop but it's not running for task: {name or 'unnamed'}"
                        )
                        return None
                except RuntimeError:
                    logger.error(
                        f"No event loop found for background task: {name or 'unnamed'}"
                    )
                    return None
            else:
                loop = _main_event_loop

    # Check if we're in the same thread as the event loop
    current_thread = threading.current_thread()

    try:
        # Try to get the loop thread if it's stored
        if hasattr(loop, '_thread_id'):
            loop_thread_id = loop._thread_id
        else:
            # Assume we're in the right thread if we can't determine
            loop_thread_id = current_thread.ident

        if current_thread.ident != loop_thread_id:
            # We're in a different thread, use run_coroutine_threadsafe
            logger.debug(
                f"Creating task from different thread for '{name or 'unnamed'}', "
                f"current={current_thread.ident}, loop={loop_thread_id}"
            )
            future = asyncio.run_coroutine_threadsafe(coro, loop)

            # Add cleanup for the future
            def cleanup(fut):
                _pending_background_tasks.discard(fut)
                try:
                    exception = fut.exception()
                    if exception:
                        logger.error(
                            f"Background task '{name}' failed: {exception}",
                            exc_info=exception,
                        )
                except asyncio.CancelledError:
                    logger.debug(f"Background task '{name}' was cancelled")
                except Exception as e:
                    logger.error(f"Error checking background task status: {e}")

            future.add_done_callback(cleanup)
            _pending_background_tasks.add(future)
            logger.debug(
                f"Created thread-safe background future: {name or 'unnamed'}, "
                f"total pending: {len(_pending_background_tasks)}"
            )
            return future
        else:
            # We're in the same thread, use create_task
            if loop.is_closed():
                logger.warning(
                    f"Event loop is closed, cannot create background task: {name or 'unnamed'}"
                )
                return None

            task = asyncio.create_task(coro, name=name)
            _pending_background_tasks.add(task)

            # Remove task from set when it completes to prevent memory leak
            def cleanup(task_ref):
                _pending_background_tasks.discard(task_ref)
                try:
                    # Check if the task raised an exception
                    exception = task_ref.exception()
                    if exception:
                        logger.error(
                            f"Background task '{task_ref.get_name()}' failed: {exception}",
                            exc_info=exception,
                        )
                except asyncio.CancelledError:
                    logger.debug(
                        f"Background task '{task_ref.get_name()}' was cancelled"
                    )
                except Exception as e:
                    logger.error(f"Error checking background task status: {e}")

            task.add_done_callback(cleanup)
            logger.debug(
                f"Created background task: {task.get_name()}, "
                f"total pending: {len(_pending_background_tasks)}"
            )
            return task
    except RuntimeError as e:
        logger.error(f"Failed to create background task '{name}': {e}")
        return None


def get_pending_task_count():
    """
    Get the number of pending background tasks.

    Returns:
        Number of active background tasks
    """
    return len(_pending_background_tasks)
