import logging
import os
import subprocess
import sys
import sysconfig
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.worker.backends.base import InferenceServer

logger = logging.getLogger(__name__)


class VLLMServer(InferenceServer):
    def start(self):
        command_path = os.path.join(sysconfig.get_path("scripts"), "vllm")
        arguments = [
            "serve",
            self._model_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(self._model_instance.port),
            "--max-model-len",
            "8192",
            "--served-model-name",
            self._model_instance.model_name,
            "--trust-remote-code",
        ]

        try:
            logger.info("Starting vllm server")
            logger.debug(f"Run vllm with arguments: {' '.join(arguments)}")
            subprocess.run(
                [command_path] + arguments,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
        except Exception as e:
            error_message = f"Failed to run the vllm server: {e}"
            logger.error(error_message)
            try:
                patch_dict = {
                    "state_message": error_message,
                    "state": ModelInstanceStateEnum.ERROR,
                }
                self._update_model_instance(self._model_instance.id, **patch_dict)
            except Exception as ue:
                logger.error(f"Failed to update model instance: {ue}")
