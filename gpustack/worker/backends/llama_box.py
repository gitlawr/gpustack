import logging
import platform
import subprocess
import sys
from typing import Optional
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.utils.command import get_platform_command
from gpustack.worker.backends.base import InferenceServer
from gpustack.utils.compat_importlib import pkg_resources

logger = logging.getLogger(__name__)


class LlamaBoxServer(InferenceServer):
    def start(self):
        command_path = pkg_resources.files(
            "gpustack.third_party.bin.llama-box"
        ).joinpath(self._get_command())

        layers = -1
        claim = self._model_instance.computed_resource_claim
        if claim is not None and claim.get("offload_layers") is not None:
            layers = claim.get("offload_layers")

        env = self._get_env(self._model_instance.gpu_index)

        arguments = [
            "--host",
            "0.0.0.0",
            "--embeddings",
            "--gpu-layers",
            str(layers),
            "--parallel",
            "4",
            "--ctx-size",
            "8192",
            "--port",
            str(self._model_instance.port),
            "--model",
            self._model_path,
        ]

        try:
            logger.info("Starting llama-box server")
            logger.debug(f"Run llama-box with arguments: {' '.join(arguments)}")
            subprocess.run(
                [command_path] + arguments,
                stdout=sys.stdout,
                stderr=sys.stderr,
                env=env,
            )
        except Exception as e:
            error_message = f"Failed to run the llama-box server: {e}"
            logger.error(error_message)
            try:
                patch_dict = {
                    "state_message": error_message,
                    "state": ModelInstanceStateEnum.ERROR,
                }
                self._update_model_instance(self._model_instance.id, **patch_dict)
            except Exception as ue:
                logger.error(f"Failed to update model instance: {ue}")

    def _get_env(self, gpu_index: Optional[int] = None):
        index = gpu_index or 0
        system = platform.system()

        if system == "Darwin":
            return None
        elif system == "Linux":
            return {"CUDA_VISIBLE_DEVICES": str(index)}
        else:
            # TODO: support more.
            return None

    def _get_command(self):
        command_map = {
            ("Windows", "amd64"): "llama-box-windows-amd64-cuda-12.5.exe",
            ("Darwin", "amd64"): "llama-box-darwin-amd64-metal",
            ("Darwin", "arm64"): "llama-box-darwin-arm64-metal",
            ("Linux", "amd64"): "llama-box-linux-amd64-cuda-12.5",
        }

        command = get_platform_command(command_map)
        if command == "":
            raise Exception(
                f"No supported llama-box command found "
                f"for {platform.system()} {platform.machine()}."
            )
        return command
