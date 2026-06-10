# evaluation/webarena/docker/workers.py
#
# Lightweight SSH-based worker pool client for the multi-docker orchestrator.
# Ported from project/PTE/eval/docker/workers_new.py — GLPAT and health-check
# logic removed since this project drives agents via env vars, not URL params.
#
# Orchestrator: /scr2/webagent-verified/webarena_orchestrator/orchestrator.py
# Requires REMOTE_HOST env var (e.g. annabella@red5k.cs.berkeley.edu).

import json
import os
import subprocess
from typing import Optional

ORCH = '/scr2/webagent-verified/webarena_orchestrator/orchestrator.py'

# Maps server name → the URL field returned by the orchestrator's acquire command.
_URL_FIELD = {
    'gitlab': 'gitlab_url',
    'shopping': 'shopping_url',
    'reddit': 'reddit_url',
}

# Maps orchestrator URL field → os.environ key used by utils.py
FIELD_TO_ENV = {
    'gitlab_url': 'GITLAB',
    'shopping_url': 'SHOPPING',
    'reddit_url': 'REDDIT',
}


def _server() -> str:
    host = os.environ.get('REMOTE_HOST')
    if not host:
        raise RuntimeError(
            'REMOTE_HOST not set — export REMOTE_HOST=user@host before using --multi-docker'
        )
    return host


def num_workers() -> int:
    result = subprocess.run(
        ['ssh', _server(), f'python3 {ORCH} num_workers'],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f'num_workers exited {result.returncode}: '
            f'stdout={result.stdout.strip()!r} stderr={result.stderr.strip()!r}'
        )
    return int(result.stdout.strip())


def acquire_worker(task_id: str) -> dict:
    result = subprocess.run(
        ['ssh', _server(), f'python3 {ORCH} acquire --task-id {task_id}'],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f'acquire_worker exited {result.returncode}: '
            f'stdout={result.stdout.strip()!r} stderr={result.stderr.strip()!r}'
        )
    data = json.loads(result.stdout)
    if 'error' in data:
        raise RuntimeError(f"No available workers: {data['error']}")
    return data


def release_worker(
    worker_id: int,
    read_only: bool = False,
    force_restart: Optional[bool] = None,
    server: Optional[str] = None,
) -> None:
    cmd = f'python3 {ORCH} release --worker-id {worker_id}'
    if force_restart is not None:
        if not force_restart:
            cmd += ' --read-only'
    elif read_only:
        cmd += ' --read-only'
    elif server is not None:
        cmd += f' --server {server}'
    import logging as _logging

    result = subprocess.run(
        ['ssh', _server(), cmd], check=False, timeout=60, capture_output=True, text=True
    )
    if result.returncode != 0:
        _logging.warning(
            f'release_worker({worker_id}) exited {result.returncode}: '
            f'stdout={result.stdout.strip()!r} stderr={result.stderr.strip()!r}'
        )


def server_urls_for_worker(worker: dict) -> dict:
    """Return {ENV_VAR: url} for every service that is enabled on this worker."""
    return {
        FIELD_TO_ENV[field]: worker[field]
        for field in _URL_FIELD.values()
        if worker.get(field) and field in FIELD_TO_ENV
    }
