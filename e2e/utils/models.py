"""Model operation helpers for E2E testing."""

import logging

from .client import GPUStackClient
from .wait import wait_for_model_ready, wait_for_model_deleted

logger = logging.getLogger(__name__)


class ModelHelper:
    """Model operation helper class."""

    def __init__(self, client: GPUStackClient):
        """
        Initialize helper.

        Args:
            client: GPUStack client
        """
        self.client = client
        self._default_cluster_id = None

    def _get_default_cluster_id(self) -> int:
        """Get the default cluster ID, cached after first call."""
        if self._default_cluster_id is None:
            cluster = self.client.get_default_cluster()
            if cluster is None:
                raise RuntimeError(
                    "No default cluster found. "
                    "Ensure the server is fully initialized before creating models."
                )
            self._default_cluster_id = cluster["id"]
            logger.info(f"Using default cluster ID: {self._default_cluster_id}")
        return self._default_cluster_id

    def deploy_model_from_catalog(
        self,
        catalog_id: str,
        backend: str = "vLLM",
        name: str = None,
        replicas: int = 1,
        wait: bool = True,
        timeout: int = 600,
        **kwargs,
    ) -> dict:
        """
        Deploy model from catalog.

        Args:
            catalog_id: Catalog model ID
            backend: Inference backend
            name: Model name (defaults to catalog name)
            replicas: Number of replicas
            wait: Wait for model to be ready
            timeout: Wait timeout in seconds
            **kwargs: Additional model parameters

        Returns:
            Model info
        """
        # Get catalog model info
        catalog_model = self.client.get_catalog_model(catalog_id)
        logger.info(f"Found catalog model: {catalog_model.get('name', catalog_id)}")

        # Determine model name
        if not name:
            name = catalog_model.get("name", catalog_id)

        # Create model
        model_data = {
            "name": name,
            "source": catalog_model.get("source", "huggingface"),
            "replicas": replicas,
            "backend": backend,
            "categories": catalog_model.get("categories", ["llm"]),
            "cluster_id": self._get_default_cluster_id(),
        }

        # Get model source info from catalog
        if catalog_model.get("huggingface_repo_id"):
            model_data["huggingface_repo_id"] = catalog_model["huggingface_repo_id"]
        if catalog_model.get("huggingface_filename"):
            model_data["huggingface_filename"] = catalog_model["huggingface_filename"]
        if catalog_model.get("model_scope_model_id"):
            model_data["model_scope_model_id"] = catalog_model["model_scope_model_id"]

        model_data.update(kwargs)

        logger.info(f"Creating model '{name}' from catalog '{catalog_id}'")
        model = self.client.create_model(**model_data)
        model_id = model["id"]
        logger.info(f"Model created with ID: {model_id}")

        if wait:
            model = wait_for_model_ready(self.client, model_id, timeout=timeout)

        return model

    def deploy_huggingface_model(
        self,
        repo_id: str,
        name: str = None,
        filename: str = None,
        backend: str = "vLLM",
        replicas: int = 1,
        categories: list = None,
        wait: bool = True,
        timeout: int = 600,
        **kwargs,
    ) -> dict:
        """
        Deploy HuggingFace model.

        Args:
            repo_id: HuggingFace repository ID
            name: Model name (defaults to repo name)
            filename: GGUF filename (for GGUF models)
            backend: Inference backend
            replicas: Number of replicas
            categories: Model categories
            wait: Wait for model to be ready
            timeout: Wait timeout in seconds
            **kwargs: Additional model parameters

        Returns:
            Model info
        """
        if not name:
            name = repo_id.split("/")[-1]

        if not categories:
            categories = ["llm"]

        model_data = {
            "name": name,
            "source": "huggingface",
            "huggingface_repo_id": repo_id,
            "backend": backend,
            "replicas": replicas,
            "categories": categories,
            "cluster_id": self._get_default_cluster_id(),
        }

        if filename:
            model_data["huggingface_filename"] = filename

        model_data.update(kwargs)

        logger.info(f"Deploying HuggingFace model: {repo_id}")
        model = self.client.create_model(**model_data)
        model_id = model["id"]
        logger.info(f"Model created with ID: {model_id}")

        if wait:
            model = wait_for_model_ready(self.client, model_id, timeout=timeout)

        return model

    def delete_model_and_wait(
        self,
        model_id: int,
        timeout: int = 120,
    ) -> bool:
        """
        Delete model and wait for completion.

        Args:
            model_id: Model ID
            timeout: Wait timeout in seconds

        Returns:
            Whether deletion succeeded
        """
        logger.info(f"Deleting model: {model_id}")
        self.client.delete_model(model_id)
        return wait_for_model_deleted(self.client, model_id, timeout=timeout)

    def verify_model_inference(
        self,
        model_name: str,
        prompt: str = "Hello, how are you?",
        max_tokens: int = 50,
    ) -> dict:
        """
        Verify model inference.

        Args:
            model_name: Model name (or route name)
            prompt: Test prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Inference response
        """
        logger.info(f"Testing inference for model: {model_name}")

        response = self.client.chat_completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )

        # Verify response
        assert "choices" in response, "Response missing 'choices'"
        assert len(response["choices"]) > 0, "Response has no choices"
        assert "message" in response["choices"][0], "Choice missing 'message'"
        assert (
            "content" in response["choices"][0]["message"]
        ), "Message missing 'content'"

        content = response["choices"][0]["message"]["content"]
        logger.info(f"Inference successful, response length: {len(content)}")

        return response

    def verify_embedding(
        self,
        model_name: str,
        input_text: str = "Hello, world!",
    ) -> dict:
        """
        Verify embedding model.

        Args:
            model_name: Model name
            input_text: Input text

        Returns:
            Embedding response
        """
        logger.info(f"Testing embedding for model: {model_name}")

        response = self.client.embedding(model=model_name, input=input_text)

        assert "data" in response, "Response missing 'data'"
        assert len(response["data"]) > 0, "Response has no data"
        assert "embedding" in response["data"][0], "Data missing 'embedding'"

        embedding = response["data"][0]["embedding"]
        logger.info(f"Embedding successful, dimension: {len(embedding)}")

        return response

    def scale_model(
        self,
        model_id: int,
        replicas: int,
        wait: bool = True,
        timeout: int = 600,
    ) -> dict:
        """
        Scale model replicas.

        Args:
            model_id: Model ID
            replicas: Target replica count
            wait: Wait for completion
            timeout: Wait timeout in seconds

        Returns:
            Updated model info
        """
        logger.info(f"Scaling model {model_id} to {replicas} replicas")

        model = self.client.update_model(model_id, replicas=replicas)

        if wait:
            model = wait_for_model_ready(self.client, model_id, timeout=timeout)

        return model

    def get_model_instances(self, model_id: int) -> list:
        """
        Get all instances of a model.

        Args:
            model_id: Model ID

        Returns:
            Instance list
        """
        result = self.client.list_model_instances(model_id=model_id)
        return result.get("items", [])

    def get_instance_logs(self, instance_id: int, tail: int = 100) -> str:
        """
        Get instance logs.

        Args:
            instance_id: Instance ID
            tail: Number of log lines

        Returns:
            Log content
        """
        return self.client.get_model_instance_logs(instance_id, tail=tail)

    def cleanup_test_models(self, name_prefix: str = "e2e-test") -> int:
        """
        Clean up test models.

        Args:
            name_prefix: Model name prefix

        Returns:
            Number of deleted models
        """
        logger.info(f"Cleaning up test models with prefix: {name_prefix}")

        result = self.client.list_models(search=name_prefix)
        models = result.get("items", [])
        count = 0

        for model in models:
            if model.get("name", "").startswith(name_prefix):
                try:
                    self.client.delete_model(model["id"])
                    count += 1
                    logger.info(f"Deleted model: {model['name']}")
                except Exception as e:
                    logger.warning(f"Failed to delete model {model['name']}: {e}")

        return count
