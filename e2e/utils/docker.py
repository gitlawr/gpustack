"""Docker utilities for E2E testing."""

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DockerError(Exception):
    """Docker operation error."""

    pass


@dataclass
class ContainerInfo:
    """Container information."""

    id: str
    name: str
    image: str
    status: str


class DockerManager:
    """Docker manager for deployment testing."""

    def __init__(
        self,
        image: str = "gpustack/gpustack:dev",
        container_prefix: str = "gpustack-e2e",
        cache_dir: str = "/tmp/gpustack-e2e/cache",
        runtime: str = "nvidia",
        use_sudo: bool = False,
    ):
        """
        Initialize Docker manager.

        Args:
            image: GPUStack image
            container_prefix: Container name prefix
            cache_dir: Cache directory path
            runtime: GPU runtime (nvidia, or empty string to disable)
            use_sudo: Whether to use sudo for docker commands
        """
        self.image = image
        self.container_prefix = container_prefix
        self.cache_dir = Path(cache_dir)
        self.runtime = runtime
        self.use_sudo = use_sudo

        self._check_docker()

    def _check_docker(self):
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise DockerError(f"Docker not available: {result.stderr}")
        except FileNotFoundError:
            raise DockerError("Docker not installed")
        except subprocess.TimeoutExpired:
            raise DockerError("Docker command timed out")

    def _run_command(
        self,
        args: list,
        timeout: int = 60,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run a Docker command."""
        if self.use_sudo:
            cmd = ["sudo", "docker"] + args
        else:
            cmd = ["docker"] + args

        logger.debug(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if check and result.returncode != 0:
            raise DockerError(f"Docker command failed: {result.stderr}")

        return result

    def _get_container_name(self, role: str = "server") -> str:
        """Get container name with prefix."""
        return f"{self.container_prefix}-{role}"

    def pull_image(self, image: str = None) -> None:
        """Pull Docker image."""
        image = image or self.image
        logger.info(f"Pulling image: {image}")
        self._run_command(["pull", image], timeout=600)

    def run_allinone(
        self,
        container_name: str = None,
        image: str = None,
        bootstrap_password: str = "Admin@123",
        cache_dir: str = None,
        runtime: str = None,
        privileged: bool = True,
        mount_docker_sock: bool = True,
        debug: bool = True,
        disable_update_check: bool = True,
        extra_args: list = None,
    ) -> str:
        """
        Start GPUStack in all-in-one mode (Server + Worker).

        Equivalent to:
        docker run -d --name <name> \\
            --restart=unless-stopped \\
            --network host \\
            --privileged \\
            --runtime nvidia \\
            --volume /var/run/docker.sock:/var/run/docker.sock \\
            --volume <cache_dir>:/var/lib/gpustack/cache \\
            <image> --bootstrap-password <pwd> --enable-worker --debug --disable-update-check

        Args:
            container_name: Container name
            image: Docker image
            bootstrap_password: Admin password
            cache_dir: Cache directory path
            runtime: GPU runtime
            privileged: Enable privileged mode
            mount_docker_sock: Mount docker.sock
            debug: Enable debug mode
            disable_update_check: Disable update check
            extra_args: Extra GPUStack arguments

        Returns:
            Container ID
        """
        container_name = container_name or self._get_container_name("allinone")
        image = image or self.image
        cache_dir = cache_dir or str(self.cache_dir)
        runtime = runtime if runtime is not None else self.runtime

        # Ensure cache directory exists
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            "run",
            "-d",
            "--name",
            container_name,
            "--restart=unless-stopped",
            "--network",
            "host",
        ]

        if privileged:
            cmd.append("--privileged")

        if runtime:
            cmd.extend(["--runtime", runtime])

        if mount_docker_sock:
            cmd.extend(["--volume", "/var/run/docker.sock:/var/run/docker.sock"])

        # Mount cache directory
        cmd.extend(["--volume", f"{cache_dir}:/var/lib/gpustack/cache"])

        # Image
        cmd.append(image)

        # GPUStack arguments
        cmd.extend(["--bootstrap-password", bootstrap_password])
        cmd.append("--enable-worker")

        if debug:
            cmd.append("--debug")

        if disable_update_check:
            cmd.append("--disable-update-check")

        if extra_args:
            cmd.extend(extra_args)

        logger.info(f"Starting GPUStack all-in-one container: {container_name}")
        logger.info(f"Image: {image}")
        logger.debug(f"Full command: docker {' '.join(cmd)}")

        result = self._run_command(cmd, timeout=120)
        container_id = result.stdout.strip()
        logger.info(f"Container started: {container_id[:12]}")
        return container_id

    def run_server_only(
        self,
        container_name: str = None,
        image: str = None,
        bootstrap_password: str = "Admin@123",
        cache_dir: str = None,
        debug: bool = True,
        disable_update_check: bool = True,
        extra_args: list = None,
    ) -> str:
        """
        Start GPUStack in server-only mode (without Worker).

        Equivalent to:
        docker run -d --name <name> \\
            --restart=unless-stopped \\
            --network host \\
            --volume <cache_dir>:/var/lib/gpustack/cache \\
            <image> --bootstrap-password <pwd> --debug --disable-update-check

        Returns:
            Container ID
        """
        container_name = container_name or self._get_container_name("server")
        image = image or self.image
        cache_dir = cache_dir or str(self.cache_dir)

        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        cmd = [
            "run",
            "-d",
            "--name",
            container_name,
            "--restart=unless-stopped",
            "--network",
            "host",
            "--volume",
            f"{cache_dir}:/var/lib/gpustack/cache",
            image,
            "--bootstrap-password",
            bootstrap_password,
        ]

        if debug:
            cmd.append("--debug")

        if disable_update_check:
            cmd.append("--disable-update-check")

        if extra_args:
            cmd.extend(extra_args)

        logger.info(f"Starting GPUStack server-only container: {container_name}")
        logger.info(f"Image: {image}")

        result = self._run_command(cmd, timeout=120)
        container_id = result.stdout.strip()
        logger.info(f"Container started: {container_id[:12]}")
        return container_id

    def run_worker(
        self,
        server_url: str,
        token: str,
        container_name: str = None,
        worker_name: str = None,
        image: str = None,
        cache_dir: str = None,
        runtime: str = None,
        privileged: bool = True,
        mount_docker_sock: bool = True,
        debug: bool = True,
        extra_args: list = None,
    ) -> str:
        """
        Start GPUStack Worker container.

        Equivalent to:
        docker run -d --name <name> \\
            --restart=unless-stopped \\
            --network host \\
            --privileged \\
            --runtime nvidia \\
            --volume /var/run/docker.sock:/var/run/docker.sock \\
            --volume <cache_dir>:/var/lib/gpustack/cache \\
            <image> --server-url <url> --token <token> --debug

        Returns:
            Container ID
        """
        worker_name = worker_name or "worker-1"
        container_name = container_name or self._get_container_name(
            f"worker-{worker_name}"
        )
        image = image or self.image
        cache_dir = cache_dir or str(self.cache_dir / f"worker-{worker_name}")
        runtime = runtime if runtime is not None else self.runtime

        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        cmd = [
            "run",
            "-d",
            "--name",
            container_name,
            "--restart=unless-stopped",
            "--network",
            "host",
        ]

        if privileged:
            cmd.append("--privileged")

        if runtime:
            cmd.extend(["--runtime", runtime])

        if mount_docker_sock:
            cmd.extend(["--volume", "/var/run/docker.sock:/var/run/docker.sock"])

        cmd.extend(["--volume", f"{cache_dir}:/var/lib/gpustack/cache"])

        cmd.append(image)

        cmd.extend(
            [
                "--server-url",
                server_url,
                "--token",
                token,
            ]
        )

        if debug:
            cmd.append("--debug")

        if extra_args:
            cmd.extend(extra_args)

        logger.info(f"Starting GPUStack worker container: {container_name}")

        result = self._run_command(cmd, timeout=120)
        container_id = result.stdout.strip()
        logger.info(f"Worker container started: {container_id[:12]}")
        return container_id

    def stop_container(self, container_name: str, timeout: int = 30) -> None:
        """Stop a container."""
        logger.info(f"Stopping container: {container_name}")
        self._run_command(["stop", "-t", str(timeout), container_name], check=False)

    def remove_container(self, container_name: str, force: bool = True) -> None:
        """Remove a container."""
        logger.info(f"Removing container: {container_name}")
        cmd = ["rm"]
        if force:
            cmd.append("-f")
        cmd.append(container_name)
        self._run_command(cmd, check=False)

    def restart_container(self, container_name: str) -> None:
        """Restart a container."""
        logger.info(f"Restarting container: {container_name}")
        self._run_command(["restart", container_name])

    def get_container_logs(self, container_name: str, tail: int = 100) -> str:
        """Get container logs."""
        result = self._run_command(
            ["logs", "--tail", str(tail), container_name],
            check=False,
        )
        return result.stdout + result.stderr

    def get_container_status(self, container_name: str) -> Optional[str]:
        """Get container status."""
        result = self._run_command(
            ["inspect", "-f", "{{.State.Status}}", container_name],
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    def is_container_running(self, container_name: str) -> bool:
        """Check if container is running."""
        return self.get_container_status(container_name) == "running"

    def wait_for_container_running(
        self,
        container_name: str,
        timeout: int = 60,
        interval: int = 2,
    ) -> bool:
        """Wait for container to be running."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.is_container_running(container_name):
                logger.info(f"Container {container_name} is running")
                return True
            time.sleep(interval)

        raise DockerError(f"Timeout waiting for container {container_name} to run")

    def cleanup_by_name(self, container_name: str) -> None:
        """Clean up a specific container."""
        self.remove_container(container_name, force=True)

    def cleanup_all(self) -> None:
        """Clean up all e2e test containers."""
        logger.info("Cleaning up all e2e containers")

        result = self._run_command(
            [
                "ps",
                "-a",
                "--filter",
                f"name={self.container_prefix}",
                "--format",
                "{{.Names}}",
            ],
            check=False,
        )

        containers = result.stdout.strip().split("\n")
        for container in containers:
            if container:
                self.remove_container(container, force=True)

    def cleanup_cache(self) -> None:
        """Clean up cache directory."""
        if self.cache_dir.exists():
            logger.info(f"Removing cache directory: {self.cache_dir}")
            shutil.rmtree(self.cache_dir, ignore_errors=True)

    def exec_command(
        self,
        container_name: str,
        command: list,
        timeout: int = 60,
    ) -> subprocess.CompletedProcess:
        """Execute command in container."""
        cmd = ["exec", container_name] + command
        return self._run_command(cmd, timeout=timeout)

    # ==================== Convenience Methods ====================

    def deploy(self, mode: str = "allinone", **kwargs) -> str:
        """
        Deploy GPUStack.

        Args:
            mode: Deployment mode (allinone, server_only)
            **kwargs: Arguments passed to the corresponding method

        Returns:
            Container ID
        """
        if mode == "allinone":
            return self.run_allinone(**kwargs)
        elif mode == "server_only":
            return self.run_server_only(**kwargs)
        else:
            raise ValueError(f"Unknown deployment mode: {mode}")

    def get_server_url(self, host: str = "localhost", port: int = 80) -> str:
        """Get server URL (for host network mode)."""
        return f"http://{host}:{port}"
