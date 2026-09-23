"""Every log a worker serves has something that writes it.

A log endpoint and the code that produces its lines live in different files,
each with tests that pass on their own. When a merge kept one and dropped the
other, every cache service instance reported no logs at all and nothing
failed -- the reader was correct about a directory nobody filled.
"""

import ast
import pathlib
import re


def _worker_log_endpoints():
    """The routes a worker serves logs from, by how they get the lines."""
    src = pathlib.Path("gpustack/routes/worker/logs.py").read_text()
    tree = ast.parse(src)
    endpoints = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(
            isinstance(d, ast.Call) and "router" in ast.unparse(d.func)
            for d in node.decorator_list
        ):
            continue
        endpoints[node.name] = ast.unparse(node)
    return endpoints


def _log_subdirectories(body: str):
    """The log subdirectories an endpoint reads files out of."""
    return set(re.findall(r"log_dir\)?\s*/\s*['\"]([a-z-]+)['\"]", body))


def _written_subdirectories():
    """The log subdirectories the worker writes into."""
    written = set()
    for path in pathlib.Path("gpustack/worker").rglob("*.py"):
        written |= set(re.findall(r"log_dir\}/([a-z-]+)", path.read_text()))
        written |= set(
            re.findall(r"log_dir\)?\s*/\s*['\"]([a-z-]+)['\"]", path.read_text())
        )
    return written


def test_every_log_directory_read_is_one_the_worker_writes():
    written = _written_subdirectories()

    unwritten = {}
    for name, body in _worker_log_endpoints().items():
        missing = _log_subdirectories(body) - written
        if missing:
            unwritten[name] = sorted(missing)

    assert not unwritten, (
        f"these endpoints read log directories nothing writes: {unwritten}; "
        f"the worker writes {sorted(written)}"
    )


def test_the_scan_sees_the_directories_it_is_meant_to():
    """Guard on the guard: a regex that matched nothing would make the check
    above pass for every endpoint, including a broken one."""
    written = _written_subdirectories()

    assert "serve" in written
    assert "benchmarks" in written
