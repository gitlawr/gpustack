"""
Extension plugin interface for GPUStack.

Third-party or enterprise plugins can implement this interface
and register via the ``gpustack.plugins`` entry-point group.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generator, Optional, Tuple

from fastapi import FastAPI

from gpustack.config.config import Config

if TYPE_CHECKING:
    from gpustack.server.coordinator import Coordinator

logger = logging.getLogger(__name__)


def iter_plugin_classes() -> Generator[Tuple[str, type], None, None]:
    """
    Iterate over all registered plugin classes.

    Yields:
        Tuple of (plugin_name, plugin_class)
    """
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group="gpustack.plugins"):
            try:
                plugin_class = ep.load()
                yield ep.name, plugin_class
            except Exception:
                logger.warning(f"Failed to load plugin class: {ep.name}", exc_info=True)
    except ImportError:
        pass


_plugin_cache: Optional[list] = None


def iter_plugins() -> Generator[Tuple[str, "Plugin"], None, None]:
    """
    Iterate over all registered plugin instances.

    Instances are cached so repeated calls return the same objects.

    Yields:
        Tuple of (plugin_name, plugin_instance)
    """
    global _plugin_cache
    if _plugin_cache is None:
        _plugin_cache = []
        for name, plugin_class in iter_plugin_classes():
            try:
                plugin = plugin_class()
                if isinstance(plugin, Plugin):
                    _plugin_cache.append((name, plugin))
                else:
                    logger.warning(f"Plugin {name} does not implement Plugin interface")
            except Exception:
                logger.warning(f"Failed to instantiate plugin: {name}", exc_info=True)
    yield from _plugin_cache


class Plugin(ABC):
    """Base class that all extension plugins must implement."""

    @abstractmethod
    def register(self, app: FastAPI, cfg: Config) -> None:
        """
        Register the plugin with the FastAPI application.

        This method is called after the FastAPI app is created.

        Args:
            app: The FastAPI application instance
            cfg: GPUStack configuration
        """

    async def create_coordinator(self, cfg: Config) -> Optional["Coordinator"]:
        """
        Create a coordinator for distributed mode.

        Args:
            cfg: GPUStack configuration

        Returns:
            A Coordinator instance or None to use default local coordinator
        """
        return None

    def setup_start_cmd(self, parser) -> None:
        """
        Set up CLI arguments for the 'start' command.

        This method is called when setting up the 'gpustack start' command.
        Plugins can add their own command-line flags here.
        Arguments are automatically parsed and available via the config object.

        Args:
            parser: The argparse ArgumentParser for the 'start' command
        """
        pass
