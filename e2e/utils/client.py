"""GPUStack API client for E2E testing."""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class PaginatedResponse:
    """Paginated response."""

    items: list
    page: int
    per_page: int
    total: int
    total_page: int


class GPUStackClientError(Exception):
    """GPUStack client error."""

    def __init__(self, message: str, status_code: int = 0, response: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class GPUStackClient:
    """GPUStack API client."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        admin_password: str = "",
        timeout: int = 30,
        verify_ssl: bool = True,
    ):
        """
        Initialize client.

        Args:
            base_url: GPUStack server URL
            api_key: API key (takes priority)
            admin_password: Admin password (used for login if no API key)
            timeout: Request timeout (seconds)
            verify_ssl: Verify SSL certificate
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.admin_password = admin_password
        self.timeout = timeout
        self.verify_ssl = verify_ssl

        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            verify=verify_ssl,
        )
        self._session_cookie: Optional[str] = None

        # If no API key, try to login
        if not self.api_key and self.admin_password:
            self._login()

    def _login(self):
        """Login with password to get session."""
        response = self._client.post(
            "/auth/login",
            data={"username": "admin", "password": self.admin_password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if response.status_code != 200:
            raise GPUStackClientError(
                f"Login failed ({self.base_url}): {response.text}",
                status_code=response.status_code,
                response=response,
            )
        self._session_cookie = response.cookies.get("gpustack_session")
        logger.info("Successfully logged in to GPUStack")

    def _get_headers(self) -> dict:
        """Get request headers."""
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        json: Optional[dict] = None,
        data: Optional[dict] = None,
        **kwargs,
    ) -> httpx.Response:
        """Send request."""
        url = path if path.startswith("http") else path
        headers = self._get_headers()
        headers.update(kwargs.pop("headers", {}))

        response = self._client.request(
            method,
            url,
            params=params,
            json=json,
            data=data,
            headers=headers,
            **kwargs,
        )

        if response.status_code >= 400:
            raise GPUStackClientError(
                f"Request failed: {method} {url} - {response.status_code}: {response.text}",
                status_code=response.status_code,
                response=response,
            )

        return response

    def _get(self, path: str, **kwargs) -> Any:
        """GET request."""
        return self._request("GET", path, **kwargs).json()

    def _post(self, path: str, **kwargs) -> Any:
        """POST request."""
        return self._request("POST", path, **kwargs).json()

    def _put(self, path: str, **kwargs) -> Any:
        """PUT request."""
        return self._request("PUT", path, **kwargs).json()

    def _delete(self, path: str, **kwargs) -> Optional[Any]:
        """DELETE request."""
        response = self._request("DELETE", path, **kwargs)
        if response.text:
            return response.json()
        return None

    def close(self):
        """Close client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== Health & Version ====================

    def health_check(self) -> bool:
        """Health check."""
        try:
            response = self._client.get("/healthz")
            return response.status_code == 200
        except Exception:
            return False

    def ready_check(self) -> bool:
        """Ready check."""
        try:
            response = self._client.get("/readyz")
            return response.status_code == 200
        except Exception:
            return False

    def get_version(self) -> dict:
        """Get version info."""
        return self._get("/version")

    # ==================== Auth ====================

    def get_auth_config(self) -> dict:
        """Get auth configuration."""
        return self._get("/auth/config")

    def login(self, username: str, password: str) -> bool:
        """Login."""
        response = self._client.post(
            "/auth/login",
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        return response.status_code == 200

    def update_password(self, current_password: str, new_password: str) -> dict:
        """Update password."""
        return self._post(
            "/auth/update-password",
            json={
                "current_password": current_password,
                "new_password": new_password,
            },
        )

    # ==================== User Management ====================

    def list_users(self, page: int = 1, per_page: int = 100, **kwargs) -> dict:
        """List users."""
        params = {"page": page, "perPage": per_page, **kwargs}
        return self._get("/v2/users", params=params)

    def get_user(self, user_id: int) -> dict:
        """Get user by ID."""
        return self._get(f"/v2/users/{user_id}")

    def get_current_user(self) -> dict:
        """Get current user."""
        return self._get("/v2/users/me")

    def create_user(
        self,
        username: str,
        password: str,
        is_admin: bool = False,
        full_name: str = "",
    ) -> dict:
        """Create user."""
        return self._post(
            "/v2/users",
            json={
                "username": username,
                "password": password,
                "is_admin": is_admin,
                "full_name": full_name,
            },
        )

    def delete_user(self, user_id: int) -> None:
        """Delete user."""
        self._delete(f"/v2/users/{user_id}")

    # ==================== API Key Management ====================

    def list_api_keys(self, user_id: str = "") -> dict:
        """List API keys."""
        params = {}
        if user_id:
            params["user_id"] = user_id
        return self._get("/v2/api-keys", params=params)

    def create_api_key(
        self,
        name: str,
        description: str = "",
        expires_in: Optional[int] = None,
        scope: list = None,
    ) -> dict:
        """Create API key."""
        data = {"name": name, "description": description}
        if expires_in:
            data["expires_in"] = expires_in
        if scope:
            data["scope"] = scope
        return self._post("/v2/api-keys", json=data)

    def delete_api_key(self, key_id: int) -> None:
        """Delete API key."""
        self._delete(f"/v2/api-keys/{key_id}")

    # ==================== Model Management ====================

    def list_models(
        self,
        page: int = 1,
        per_page: int = 100,
        state: str = "",
        categories: list = None,
        cluster_id: int = None,
        **kwargs,
    ) -> dict:
        """List models."""
        params = {"page": page, "perPage": per_page, **kwargs}
        if state:
            params["state"] = state
        if categories:
            params["categories"] = categories
        if cluster_id:
            params["cluster_id"] = cluster_id
        return self._get("/v2/models", params=params)

    def get_model(self, model_id: int) -> dict:
        """Get model by ID."""
        return self._get(f"/v2/models/{model_id}")

    def get_model_by_name(self, name: str) -> Optional[dict]:
        """Get model by name."""
        result = self.list_models(search=name)
        for model in result.get("items", []):
            if model.get("name") == name:
                return model
        return None

    def create_model(
        self,
        name: str,
        source: str = "huggingface",
        huggingface_repo_id: str = "",
        replicas: int = 1,
        backend: str = "vLLM",
        categories: list = None,
        backend_parameters: list = None,
        gpu_selector: dict = None,
        worker_selector: dict = None,
        cluster_id: int = None,
        enable_model_route: bool = True,
        **kwargs,
    ) -> dict:
        """Create model."""
        data = {
            "name": name,
            "source": source,
            "replicas": replicas,
            "backend": backend,
            "enable_model_route": enable_model_route,
        }
        if huggingface_repo_id:
            data["huggingface_repo_id"] = huggingface_repo_id
        if categories:
            data["categories"] = categories
        if backend_parameters:
            data["backend_parameters"] = backend_parameters
        if gpu_selector:
            data["gpu_selector"] = gpu_selector
        if worker_selector:
            data["worker_selector"] = worker_selector
        if cluster_id:
            data["cluster_id"] = cluster_id
        data.update(kwargs)
        return self._post("/v2/models", json=data)

    def update_model(self, model_id: int, **kwargs) -> dict:
        """Update model."""
        return self._put(f"/v2/models/{model_id}", json=kwargs)

    def delete_model(self, model_id: int) -> None:
        """Delete model."""
        self._delete(f"/v2/models/{model_id}")

    # ==================== Model Instances ====================

    def list_model_instances(
        self,
        model_id: int = None,
        worker_id: int = None,
        state: str = "",
        **kwargs,
    ) -> dict:
        """List model instances."""
        params = {**kwargs}
        if model_id:
            params["model_id"] = model_id
        if worker_id:
            params["worker_id"] = worker_id
        if state:
            params["state"] = state
        return self._get("/v2/model-instances", params=params)

    def get_model_instance(self, instance_id: int) -> dict:
        """Get model instance by ID."""
        return self._get(f"/v2/model-instances/{instance_id}")

    def get_model_instance_logs(
        self,
        instance_id: int,
        tail: int = 100,
    ) -> str:
        """Get model instance logs."""
        response = self._request(
            "GET",
            f"/v2/model-instances/{instance_id}/logs",
            params={"tail": tail},
        )
        return response.text

    def delete_model_instance(self, instance_id: int) -> None:
        """Delete model instance."""
        self._delete(f"/v2/model-instances/{instance_id}")

    # ==================== Worker Management ====================

    def list_workers(
        self,
        state: str = "",
        cluster_id: int = None,
        **kwargs,
    ) -> dict:
        """List workers."""
        params = {**kwargs}
        if state:
            params["state"] = state
        if cluster_id:
            params["cluster_id"] = cluster_id
        return self._get("/v2/workers", params=params)

    def get_worker(self, worker_id: int) -> dict:
        """Get worker by ID."""
        return self._get(f"/v2/workers/{worker_id}")

    def update_worker(self, worker_id: int, **kwargs) -> dict:
        """Update worker."""
        return self._put(f"/v2/workers/{worker_id}", json=kwargs)

    def delete_worker(self, worker_id: int) -> None:
        """Delete worker."""
        self._delete(f"/v2/workers/{worker_id}")

    def set_worker_maintenance(
        self,
        worker_id: int,
        enabled: bool,
        message: str = "",
    ) -> dict:
        """Set worker maintenance mode."""
        return self.update_worker(
            worker_id,
            maintenance={"enabled": enabled, "message": message},
        )

    # ==================== Cluster Management ====================

    def list_clusters(self, **kwargs) -> dict:
        """List clusters."""
        return self._get("/v2/clusters", params=kwargs)

    def get_cluster(self, cluster_id: int) -> dict:
        """Get cluster by ID."""
        return self._get(f"/v2/clusters/{cluster_id}")

    def get_default_cluster(self) -> Optional[dict]:
        """Get default cluster."""
        result = self.list_clusters()
        for cluster in result.get("items", []):
            if cluster.get("is_default"):
                return cluster
        return None

    def create_cluster(
        self,
        name: str,
        provider: str,
        credential_id: int = None,
        **kwargs,
    ) -> dict:
        """Create cluster."""
        data = {"name": name, "provider": provider}
        if credential_id:
            data["credential_id"] = credential_id
        data.update(kwargs)
        return self._post("/v2/clusters", json=data)

    def delete_cluster(self, cluster_id: int) -> None:
        """Delete cluster."""
        self._delete(f"/v2/clusters/{cluster_id}")

    def get_registration_token(self, cluster_id: int) -> dict:
        """Get worker registration token."""
        return self._get(f"/v2/clusters/{cluster_id}/registration-token")

    def get_k8s_manifests(self, cluster_id: int) -> str:
        """Get Kubernetes deployment manifests."""
        response = self._request("GET", f"/v2/clusters/{cluster_id}/manifests")
        return response.text

    # ==================== Model Provider ====================

    def list_model_providers(self, **kwargs) -> dict:
        """List model providers."""
        return self._get("/v2/model-providers", params=kwargs)

    def get_model_provider(self, provider_id: int) -> dict:
        """Get model provider by ID."""
        return self._get(f"/v2/model-providers/{provider_id}")

    def create_model_provider(
        self,
        name: str,
        config: dict,
        models: list = None,
        api_tokens: list = None,
        **kwargs,
    ) -> dict:
        """Create model provider."""
        data = {"name": name, "config": config}
        if models:
            data["models"] = models
        if api_tokens:
            data["api_tokens"] = api_tokens
        data.update(kwargs)
        return self._post("/v2/model-providers", json=data)

    def update_model_provider(self, provider_id: int, **kwargs) -> dict:
        """Update model provider."""
        return self._put(f"/v2/model-providers/{provider_id}", json=kwargs)

    def delete_model_provider(self, provider_id: int) -> None:
        """Delete model provider."""
        self._delete(f"/v2/model-providers/{provider_id}")

    def test_provider_model(
        self,
        provider_id: int = None,
        config: dict = None,
        model_name: str = "",
        api_tokens: list = None,
    ) -> dict:
        """Test provider model connection."""
        data = {"model_name": model_name}
        if config:
            data["config"] = config
        if api_tokens:
            data["api_tokens"] = api_tokens

        if provider_id:
            return self._post(
                f"/v2/model-providers/{provider_id}/test-model", json=data
            )
        return self._post("/v2/model-providers/test-model", json=data)

    # ==================== Model Route ====================

    def list_model_routes(self, **kwargs) -> dict:
        """List model routes."""
        return self._get("/v2/model-routes", params=kwargs)

    def get_model_route(self, route_id: int) -> dict:
        """Get model route by ID."""
        return self._get(f"/v2/model-routes/{route_id}")

    def get_model_route_by_name(self, name: str) -> Optional[dict]:
        """Get model route by name."""
        result = self.list_model_routes(search=name)
        for route in result.get("items", []):
            if route.get("name") == name:
                return route
        return None

    def create_model_route(
        self,
        name: str,
        targets: list,
        categories: list = None,
        **kwargs,
    ) -> dict:
        """Create model route."""
        data = {"name": name, "targets": targets}
        if categories:
            data["categories"] = categories
        data.update(kwargs)
        return self._post("/v2/model-routes", json=data)

    def update_model_route(self, route_id: int, **kwargs) -> dict:
        """Update model route."""
        return self._put(f"/v2/model-routes/{route_id}", json=kwargs)

    def delete_model_route(self, route_id: int) -> None:
        """Delete model route."""
        self._delete(f"/v2/model-routes/{route_id}")

    def add_route_targets(self, route_id: int, targets: list) -> dict:
        """Add targets to route."""
        return self._post(
            f"/v2/model-routes/{route_id}/add-targets",
            json={"targets": targets},
        )

    # ==================== GPU Devices ====================

    def list_gpu_devices(self, **kwargs) -> dict:
        """List GPU devices."""
        return self._get("/v2/gpu-devices", params=kwargs)

    # ==================== Catalog ====================

    def list_catalog_models(self, **kwargs) -> dict:
        """List catalog models."""
        return self._get("/v2/catalogs/models", params=kwargs)

    def get_catalog_model(self, model_id: str) -> dict:
        """Get catalog model by ID."""
        return self._get(f"/v2/catalogs/models/{model_id}")

    # ==================== Benchmarks ====================

    def list_benchmarks(self, **kwargs) -> dict:
        """List benchmarks."""
        return self._get("/v2/benchmarks", params=kwargs)

    def get_benchmark(self, benchmark_id: int) -> dict:
        """Get benchmark by ID."""
        return self._get(f"/v2/benchmarks/{benchmark_id}")

    def create_benchmark(
        self,
        name: str,
        model_id: int,
        **kwargs,
    ) -> dict:
        """Create benchmark."""
        data = {"name": name, "model_id": model_id}
        data.update(kwargs)
        return self._post("/v2/benchmarks", json=data)

    def delete_benchmark(self, benchmark_id: int) -> None:
        """Delete benchmark."""
        self._delete(f"/v2/benchmarks/{benchmark_id}")

    # ==================== OpenAI Compatible API ====================

    def openai_list_models(self) -> dict:
        """List OpenAI compatible models."""
        return self._get("/v1-openai/models")

    def chat_completion(
        self,
        model: str,
        messages: list,
        stream: bool = False,
        **kwargs,
    ) -> dict:
        """Chat completion."""
        data = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        data.update(kwargs)
        return self._post("/v1-openai/chat/completions", json=data)

    def completion(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        **kwargs,
    ) -> dict:
        """Text completion."""
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }
        data.update(kwargs)
        return self._post("/v1-openai/completions", json=data)

    def embedding(self, model: str, input: str | list, **kwargs) -> dict:
        """Embedding."""
        data = {"model": model, "input": input}
        data.update(kwargs)
        return self._post("/v1-openai/embeddings", json=data)

    def audio_speech(
        self,
        model: str,
        input: str,
        voice: str = "alloy",
        **kwargs,
    ) -> bytes:
        """Text to speech."""
        data = {
            "model": model,
            "input": input,
            "voice": voice,
        }
        data.update(kwargs)
        response = self._request("POST", "/v1-openai/audio/speech", json=data)
        return response.content

    def audio_transcription(
        self,
        model: str,
        file: bytes,
        filename: str = "audio.wav",
        **kwargs,
    ) -> dict:
        """Speech to text."""
        files = {"file": (filename, file)}
        data = {"model": model}
        data.update(kwargs)
        return self._post("/v1-openai/audio/transcriptions", data=data, files=files)

    # ==================== Dashboard ====================

    def get_dashboard(self) -> dict:
        """Get dashboard data."""
        return self._get("/v2/dashboard")

    # ==================== Cloud Credentials ====================

    def list_cloud_credentials(self, **kwargs) -> dict:
        """List cloud credentials."""
        return self._get("/v2/cloud-credentials", params=kwargs)

    def create_cloud_credential(
        self,
        name: str,
        provider: str,
        config: dict,
        **kwargs,
    ) -> dict:
        """Create cloud credential."""
        data = {"name": name, "provider": provider, "config": config}
        data.update(kwargs)
        return self._post("/v2/cloud-credentials", json=data)

    def delete_cloud_credential(self, credential_id: int) -> None:
        """Delete cloud credential."""
        self._delete(f"/v2/cloud-credentials/{credential_id}")

    # ==================== Worker Pools ====================

    def list_worker_pools(self, cluster_id: int = None, **kwargs) -> dict:
        """List worker pools."""
        params = {**kwargs}
        if cluster_id:
            params["cluster_id"] = cluster_id
        return self._get("/v2/worker-pools", params=params)

    def create_worker_pool(
        self,
        cluster_id: int,
        name: str,
        replicas: int = 1,
        **kwargs,
    ) -> dict:
        """Create worker pool."""
        data = {"name": name, "replicas": replicas}
        data.update(kwargs)
        return self._post(f"/v2/clusters/{cluster_id}/worker-pools", json=data)

    def update_worker_pool(self, pool_id: int, **kwargs) -> dict:
        """Update worker pool."""
        return self._put(f"/v2/worker-pools/{pool_id}", json=kwargs)

    def delete_worker_pool(self, pool_id: int) -> None:
        """Delete worker pool."""
        self._delete(f"/v2/worker-pools/{pool_id}")
