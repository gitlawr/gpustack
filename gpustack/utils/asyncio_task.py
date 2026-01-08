import asyncio
import logging

logger = logging.getLogger(__name__)

# Global set to hold background tasks, preventing them from being garbage collected
_pending_background_tasks = set()


def create_background_task(coro, *, name: str = None):
    """
    Create an asyncio task that is protected from garbage collection.

    This is a wrapper around asyncio.create_task() that maintains a strong reference
    to the task to prevent it from being garbage collected before it runs.

    Args:
        coro: The coroutine to run as a task
        name: Optional name for the task

    Returns:
        The created Task object

    Example:
        task = create_background_task(
            event_bus.publish("topic", event),
            name="publish_modelinstance_deleted"
        )
    """
    task = asyncio.create_task(coro, name=name)
    _pending_background_tasks.add(task)
    # Remove task from set when it completes to prevent memory leak
    task.add_done_callback(_pending_background_tasks.discard)
    return task


def get_pending_task_count():
    """
    Get the number of pending background tasks.

    Returns:
        Number of active background tasks
    """
    return len(_pending_background_tasks)
