"""Wait and polling utilities for E2E testing."""

import logging
import time
from typing import Callable

from .client import GPUStackClient

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Wait timeout error."""

    pass


def wait_for_condition(
    condition: Callable[[], bool],
    timeout: int = 300,
    interval: int = 5,
    description: str = "condition",
) -> bool:
    """
    Wait for a condition to be met.

    Args:
        condition: Condition function, returns True when met
        timeout: Timeout in seconds
        interval: Check interval in seconds
        description: Condition description for logging

    Returns:
        Whether condition was met

    Raises:
        TimeoutError: Timeout waiting for condition
    """
    start_time = time.time()
    last_error = None

    while time.time() - start_time < timeout:
        try:
            if condition():
                logger.info(
                    f"Condition '{description}' met after {time.time() - start_time:.1f}s"
                )
                return True
        except Exception as e:
            last_error = e
            logger.debug(f"Condition '{description}' check failed: {e}")

        time.sleep(interval)

    elapsed = time.time() - start_time
    error_msg = f"Timeout waiting for '{description}' after {elapsed:.1f}s"
    if last_error:
        error_msg += f" (last error: {last_error})"
    raise TimeoutError(error_msg)


def wait_for_model_ready(
    client: GPUStackClient,
    model_id: int,
    timeout: int = 600,
    interval: int = 10,
) -> dict:
    """
    Wait for model to be ready.

    Args:
        client: GPUStack client
        model_id: Model ID
        timeout: Timeout in seconds
        interval: Check interval in seconds

    Returns:
        Ready model info

    Raises:
        TimeoutError: Timeout waiting for model
        Exception: Model deployment failed
    """
    start_time = time.time()
    last_state = None

    while time.time() - start_time < timeout:
        try:
            model = client.get_model(model_id)
            replicas = model.get("replicas", 1)
            ready_replicas = model.get("ready_replicas", 0)

            if last_state != (ready_replicas, replicas):
                logger.info(
                    f"Model {model_id} state: {ready_replicas}/{replicas} replicas ready"
                )
                last_state = (ready_replicas, replicas)

            if ready_replicas >= replicas:
                logger.info(f"Model {model_id} is ready")
                return model

            # Check instance state for errors
            instances = client.list_model_instances(model_id=model_id)
            for instance in instances.get("items", []):
                state = instance.get("state", "")
                if state == "error":
                    error_msg = instance.get("state_message", "Unknown error")
                    raise Exception(
                        f"Model instance {instance.get('id')} failed: {error_msg}"
                    )

        except Exception as e:
            if "failed" in str(e).lower() or "error" in str(e).lower():
                raise
            logger.debug(f"Error checking model {model_id}: {e}")

        time.sleep(interval)

    raise TimeoutError(
        f"Timeout waiting for model {model_id} to be ready after {timeout}s"
    )


def wait_for_model_instance_ready(
    client: GPUStackClient,
    instance_id: int,
    timeout: int = 600,
    interval: int = 10,
) -> dict:
    """
    Wait for model instance to be ready.

    Args:
        client: GPUStack client
        instance_id: Instance ID
        timeout: Timeout in seconds
        interval: Check interval in seconds

    Returns:
        Ready instance info

    Raises:
        TimeoutError: Timeout waiting for instance
        Exception: Instance startup failed
    """
    start_time = time.time()
    last_state = None

    while time.time() - start_time < timeout:
        try:
            instance = client.get_model_instance(instance_id)
            state = instance.get("state", "")
            progress = instance.get("download_progress", 0)

            if state != last_state:
                logger.info(
                    f"Model instance {instance_id} state: {state}"
                    + (f" (download: {progress:.1%})" if progress > 0 else "")
                )
                last_state = state

            if state == "running":
                logger.info(f"Model instance {instance_id} is running")
                return instance

            if state == "error":
                error_msg = instance.get("state_message", "Unknown error")
                raise Exception(f"Model instance {instance_id} failed: {error_msg}")

        except Exception as e:
            if "failed" in str(e).lower() or "error" in str(e).lower():
                raise
            logger.debug(f"Error checking instance {instance_id}: {e}")

        time.sleep(interval)

    raise TimeoutError(
        f"Timeout waiting for model instance {instance_id} to be ready after {timeout}s"
    )


def wait_for_worker_ready(
    client: GPUStackClient,
    worker_id: int,
    timeout: int = 300,
    interval: int = 10,
) -> dict:
    """
    Wait for worker to be ready.

    Args:
        client: GPUStack client
        worker_id: Worker ID
        timeout: Timeout in seconds
        interval: Check interval in seconds

    Returns:
        Ready worker info

    Raises:
        TimeoutError: Timeout waiting for worker
        Exception: Worker startup failed
    """
    start_time = time.time()
    last_state = None

    while time.time() - start_time < timeout:
        try:
            worker = client.get_worker(worker_id)
            state = worker.get("state", "")

            if state != last_state:
                logger.info(f"Worker {worker_id} state: {state}")
                last_state = state

            if state == "ready":
                logger.info(f"Worker {worker_id} is ready")
                return worker

            if state == "error":
                error_msg = worker.get("state_message", "Unknown error")
                raise Exception(f"Worker {worker_id} failed: {error_msg}")

        except Exception as e:
            if "failed" in str(e).lower() or "error" in str(e).lower():
                raise
            logger.debug(f"Error checking worker {worker_id}: {e}")

        time.sleep(interval)

    raise TimeoutError(
        f"Timeout waiting for worker {worker_id} to be ready after {timeout}s"
    )


def wait_for_model_deleted(
    client: GPUStackClient,
    model_id: int,
    timeout: int = 120,
    interval: int = 5,
) -> bool:
    """
    Wait for model to be deleted.

    Args:
        client: GPUStack client
        model_id: Model ID
        timeout: Timeout in seconds
        interval: Check interval in seconds

    Returns:
        Whether deletion succeeded

    Raises:
        TimeoutError: Timeout waiting for deletion
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            client.get_model(model_id)
            # Model still exists
            time.sleep(interval)
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                logger.info(f"Model {model_id} has been deleted")
                return True
            raise

    raise TimeoutError(
        f"Timeout waiting for model {model_id} to be deleted after {timeout}s"
    )


def wait_for_worker_deleted(
    client: GPUStackClient,
    worker_id: int,
    timeout: int = 120,
    interval: int = 5,
) -> bool:
    """
    Wait for worker to be deleted.

    Args:
        client: GPUStack client
        worker_id: Worker ID
        timeout: Timeout in seconds
        interval: Check interval in seconds

    Returns:
        Whether deletion succeeded

    Raises:
        TimeoutError: Timeout waiting for deletion
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            client.get_worker(worker_id)
            # Worker still exists
            time.sleep(interval)
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                logger.info(f"Worker {worker_id} has been deleted")
                return True
            raise

    raise TimeoutError(
        f"Timeout waiting for worker {worker_id} to be deleted after {timeout}s"
    )


def wait_for_server_healthy(
    client: GPUStackClient,
    timeout: int = 120,
    interval: int = 5,
) -> bool:
    """
    Wait for server to be healthy.

    Args:
        client: GPUStack client
        timeout: Timeout in seconds
        interval: Check interval in seconds

    Returns:
        Whether server is healthy

    Raises:
        TimeoutError: Timeout waiting for server
    """

    def check_health():
        return client.health_check() and client.ready_check()

    return wait_for_condition(
        check_health,
        timeout=timeout,
        interval=interval,
        description="server healthy",
    )
