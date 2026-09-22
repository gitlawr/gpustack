"""
Handing out ports on a worker.

Two workloads on one worker must not be given the same port, and a workload
that already has one should keep it: the endpoints it was reached at are
copied into places that nothing refreshes, so a restart that moved a port
strands whatever is pointing at it.

What each kind of workload does with its ports differs -- how many, what they
mean, and which ranges have to be fenced off around them -- so allocation is a
session the caller drives, and this owns only the parts that would otherwise be
written once per manager: the lock, the record of what is handed out, and the
running exclusion set.
"""

import logging
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set

from gpustack.utils import network

logger = logging.getLogger(__name__)


class PortSession:
    """
    One workload's allocation. Ports taken here are excluded from each other
    and recorded together when the session ends.
    """

    def __init__(self, port_range: str, unavailable: Set[int], host: Optional[str]):
        self._port_range = port_range
        self._unavailable = unavailable
        self._host = host
        self.taken: List[int] = []

    def exclude(self, ports: Iterable[int]):
        """Keep these out of everything taken afterwards.

        For a backend that reserves a band around a port it was given, rather
        than just the port.
        """
        self._unavailable.update(ports)

    def take(self) -> int:
        port = network.get_free_port(
            port_range=self._port_range,
            unavailable_ports=self._unavailable,
            **({"host": self._host} if self._host else {}),
        )
        self._unavailable.add(port)
        self.taken.append(port)
        return port

    def reuse(self, ports: Iterable[int]) -> bool:
        """
        Take the ports this workload already had, if they are all still free.

        Returns False when any of them is gone, leaving the session untouched
        so the caller can allocate fresh ones.
        """
        wanted = [port for port in ports if port]
        if not wanted:
            return False
        for port in wanted:
            if port in self._unavailable or not network.is_port_available(port):
                return False
        self._unavailable.update(wanted)
        self.taken.extend(wanted)
        return True


class PortAllocator:
    """
    Args:
        port_range: The range to allocate from, as configured on the worker.
        peer_ports: The ports other workloads of this kind already hold on
            this worker, excluding the one being allocated for. Without it,
            only what this process handed out is known, so a restarted worker
            can re-issue a port something is already listening on.
    """

    def __init__(
        self,
        port_range: str,
        peer_ports: Optional[Callable[[Any], Set[int]]] = None,
    ):
        self._port_range = port_range
        self._peer_ports = peer_ports
        self._lock = threading.Lock()
        self._assigned: Dict[Any, Set[int]] = {}

    @contextmanager
    def session(self, key: Any, host: Optional[str] = None) -> Iterator[PortSession]:
        """
        Allocate for one workload, recording the result on a clean exit.

        Held under a lock for the whole session: two concurrent starts would
        otherwise both pick from a snapshot that does not have the other's
        ports in it.
        """
        with self._lock:
            unavailable = {port for ports in self._assigned.values() for port in ports}
            if self._peer_ports is not None:
                try:
                    unavailable.update(self._peer_ports(key))
                except Exception as e:
                    # Better to risk a collision than to fail the start: the
                    # in-process set still rules out most of them.
                    logger.warning(f"Failed to read peer ports for {key}: {e}")

            session = PortSession(self._port_range, unavailable, host)
            yield session
            if session.taken:
                self._assigned[key] = set(session.taken)

    def release(self, key: Any):
        with self._lock:
            self._assigned.pop(key, None)

    def assigned(self, key: Any) -> Set[int]:
        with self._lock:
            return set(self._assigned.get(key, ()))

    def keys(self) -> Set[Any]:
        with self._lock:
            return set(self._assigned)
