"""Boat cloud sandboxes as a Pydantic AI workspace, built on Boat's official Python SDK (`boat-sdk`).

`BoatSandbox` supplies a [Boat](https://boat.dev) sandbox, a root Ubuntu VM, as an agent run's workspace, so tools
such as Pydantic AI Harness's `Coder`, `Shell` and `FileSystem` run there. The SDK is synchronous, so its calls run
in worker threads: the backend works on asyncio and Trio, and has no connection to close.
"""

from __future__ import annotations

import base64
import json
import os
import posixpath
import shlex
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache, partial
from typing import Any, Literal, TypeVar, cast

import anyio
import anyio.to_thread
import boat_sdk as boat
from boat_sdk import boat_helpers
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import (
    CommandResult,
    FileEntry,
    SupportsCommands,
    SupportsFilesystem,
    SupportsRealpath,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai.workspaces.protocol import validate_timeout

# Core's shell filesystem derives file operations from commands. It is private API, so `pyproject.toml` pins
# `pydantic-ai-slim` to the release range this module is tested against.
from pydantic_ai.workspaces.workspace import _ShellFilesystem  # pyright: ignore[reportPrivateUsage]

__all__ = ('BoatSandbox', 'BoatSandboxBackend', 'MachineType')

MachineType = Literal['small', 'default', 'large', 'xlarge']
"""Boat's machine sizes (`small` is 2 vCPU and 4 GB); see https://docs.boat.dev/machines."""

_T = TypeVar('_T')
_OUTPUT_LIMIT = 8 * 1024 * 1024  # Per stream; Boat's own cap for synchronous commands.
_FILE_API_LIMIT = 5 * 1024 * 1024  # The most Boat's file API moves in one call.
_KEEP_ALIVE_INTERVAL = 60.0
_READY_TIMEOUT = 120.0
# Boat sends SIGTERM, then SIGKILL about 5 seconds later, and answers once the command is gone.
_STOP_TIMEOUT = 15.0
_MISSING_CWD = f'pydantic-ai-boat-missing-cwd-{uuid.uuid4().hex}'


@cache
def _api_for(key: str) -> boat.BoatApi:
    # One client per key for the process: the SDK is thread-safe and not tied to an event loop.
    return boat.BoatApi(boat.ApiClient(boat.Configuration(access_token=key)))


def _default_api() -> boat.BoatApi:
    key = os.environ.get('BOAT_API_KEY', '').strip()
    if not key:
        raise WorkspaceUnavailableError('No Boat API key: set BOAT_API_KEY, or pass a `boat_sdk.BoatApi` as `api=`.')
    return _api_for(key)


def _status(error: BaseException) -> int | None:
    """The HTTP status of a Boat API error; the SDK leaves the attribute untyped."""
    return getattr(error, 'status', None) if isinstance(error, boat.ApiException) else None


def _unavailable(error: boat.ApiException, sandbox_id: str | None) -> WorkspaceUnavailableError | None:
    """The protocol's error for a Boat answer no retry can fix, or `None` to let `error` propagate."""
    status = _status(error)
    try:
        code = str(json.loads(getattr(error, 'body', None) or '{}').get('code', ''))
    except (ValueError, AttributeError):
        code = ''
    if status in (401, 403):
        # Only the status and code: a provider's error text can quote the rejected key.
        return WorkspaceUnavailableError(f'Boat refused the request ({status} {code}); check BOAT_API_KEY.')
    if status == 404:
        what = f'sandbox {sandbox_id!r}' if sandbox_id else 'snapshot to start from'
        return WorkspaceUnavailableError(f"Boat has no {what} (deleted?). Pass `workspace='new'` to start over.")
    if status == 409 and code in ('machine_not_running', 'sandbox_not_ready', 'snapshot_data_expired'):
        return WorkspaceUnavailableError(f'Boat sandbox {sandbox_id!r} stopped ({code}); attaching again resumes it.')
    return None


def _follow(api: boat.BoatApi, sandbox_id: str, process_id: int) -> Iterator[dict[str, Any]]:
    """`boat_helpers.follow_command`, whose frames the SDK leaves untyped."""
    follow = boat_helpers.follow_command  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return cast('Iterator[dict[str, Any]]', follow(api, sandbox_id, process_id))


def _command_line(argv: Sequence[str], cwd: str | None, env: Mapping[str, str]) -> str:
    """The string Boat runs with `bash -lc`: `argv` in `cwd` with `env` added and stdin at EOF.

    `exec` keeps Boat's login shell from running `~/.bash_logout`, which prints to the output; `env --` passes
    names a `sh` such as dash would drop. A missing `cwd` prints `_MISSING_CWD` and exits 125.
    """
    if env:
        argv = ['env', '--', *(f'{name}={value}' for name, value in env.items()), *argv]
    guard = f'cd -- {shlex.quote(cwd)} || {{ printf %s {_MISSING_CWD}; exit 125; }}; ' if cwd else ''
    return shlex.join(['exec', 'sh', '-c', f'{guard}"$@" </dev/null; exit $?', 'sh', *argv])


class BoatSandboxBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem, SupportsRealpath):
    """A [Boat](https://boat.dev) sandbox behind Pydantic AI's workspace backend protocol.

    Building one does no I/O. The first operation creates a sandbox with none of the Boat account's secrets
    (`noEnv`), or attaches to the one `ref` names, resuming it if Boat stopped it. The backend never stops or
    deletes the sandbox: whoever holds the ref decides, for example with `BoatSandbox.destroy`.

    It takes the options `BoatSandbox` documents, plus `ref`, an existing sandbox to attach to.
    """

    def __init__(
        self,
        *,
        api: boat.BoatApi | None = None,
        ref: WorkspaceRef | None = None,
        machine_type: MachineType | None = None,
        snapshot: str | None = None,
        auto_stop: int | None = None,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if ref is not None and ref.provider != 'boat':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'boat'")
        if working_dir is not None and not posixpath.isabs(working_dir):
            raise ValueError(f'working_dir must be an absolute path, got {working_dir!r}')
        self._api, self._ref = api, ref
        self._machine_type, self._snapshot, self._auto_stop = machine_type, snapshot, auto_stop
        self._working_dir, self._env = working_dir, dict(env or {})
        # Sent with every create attempt, so a retry after a lost reply gets the sandbox already created.
        self._idempotency_key = uuid.uuid4().hex
        self._lock = anyio.Lock()
        self._ready = False
        self._resolved_working_dir: str | None = None
        self._ttl: int | None = None  # The sandbox's auto-stop, which `_extend` renews; `None` when off.
        self._extended_at = -float('inf')
        self._shell = _ShellFilesystem(self)

    @property
    def ref(self) -> WorkspaceRef | None:
        """The sandbox's identity (`provider='boat'`, `id` such as `bx_h8mygsw2`), `None` until created."""
        return self._ref

    def _boat(self) -> boat.BoatApi:
        if self._api is None:
            self._api = _default_api()
        return self._api

    async def _call(self, call: Callable[[], _T], *, abandon: bool = False) -> _T:
        """Run a blocking SDK call in a worker thread, raising the protocol's errors for Boat's refusals."""
        try:
            return await anyio.to_thread.run_sync(call, abandon_on_cancel=abandon)
        except boat.ApiException as error:
            unavailable = _unavailable(error, self._ref.id if self._ref else None)
            if unavailable is None:
                raise
            raise unavailable from error

    async def _connect(self) -> str:
        """Create or attach on first use, under a lock so concurrent first use makes one sandbox."""
        async with self._lock:
            await self._call(self._prepare)
        assert self._ref is not None
        return self._ref.id

    def _prepare(self) -> None:
        # In a worker thread: a caller cancelled meanwhile still sees a create finish and its ref recorded.
        if not self._ready:
            self._acquire(self._boat())
            self._ready = True
        self._extend()

    def _acquire(self, api: boat.BoatApi) -> None:
        created = self._ref is None
        if self._ref is None:
            # Explicit `None`s keep the SDK from sending its own defaults, such as a `type` over a snapshot's.
            body: dict[str, Any] = {'noEnv': True, 'type': self._machine_type, 'from': self._snapshot}
            body.update(dict.fromkeys(('environment', 'snapshots', 'monitoring', 'failFast', 'count')))
            if self._auto_stop is not None:
                body['ttlSeconds'] = self._auto_stop or None  # Boat reads `null` as no auto-stop.
            request = boat.CreateSandboxRequest.model_validate(body)
            sandbox = api.create(idempotency_key=self._idempotency_key, create_sandbox_request=request).sandbox
            self._ref = WorkspaceRef(provider='boat', id=sandbox.id)
        state = '' if created else api.get(self._ref.id).sandbox.state
        if state in ('error', 'cancelled'):
            raise WorkspaceUnavailableError(f"Boat sandbox {self._ref.id!r} is {state}. Pass `workspace='new'`.")
        resumed = state in ('archived', 'archiving')
        if resumed:
            api.resume(self._ref.id)
        try:
            sandbox = boat_helpers.wait_until_ready(
                api, self._ref.id, timeout_seconds=_READY_TIMEOUT, poll_interval_seconds=0.5
            )
        except RuntimeError as error:  # Stopped or failed while starting.
            raise WorkspaceUnavailableError(f'Boat sandbox {self._ref.id!r} did not start: {error}') from error
        self._ttl = sandbox.ttl_seconds
        if created or resumed:
            self._extended_at = time.monotonic()  # A create or resume sets Boat's deadline itself.
        if created and self._working_dir:
            mkdir = boat.CommandRequest(command=shlex.join(['mkdir', '-p', '--', self._working_dir]))
            result = api.command(self._ref.id, mkdir).actual_instance
            if not isinstance(result, boat.CommandResponse) or result.exit_code != 0:
                detail = result.stderr.strip() if isinstance(result, boat.CommandResponse) else ''
                raise WorkspaceError(f'Could not create working_dir {self._working_dir!r}: {detail}')

    def _extend(self) -> None:
        """Move Boat's auto-stop deadline, which commands do not move, to `ttl` from now; best effort.

        At most once a minute, or per half the ttl when that is shorter. Runs in a worker thread.
        """
        ttl, now = self._ttl, time.monotonic()
        if ttl is None or self._ref is None or now - self._extended_at < min(ttl / 2, _KEEP_ALIVE_INTERVAL):
            return
        self._extended_at = now
        try:
            self._boat().update(self._ref.id, boat.UpdateSandboxRequest.model_validate({'ttlSeconds': ttl}))
        except Exception as error:
            if _status(error) in (402, 403):
                self._ttl = None  # Refused for good, as for a key without `sandbox.update`: stop asking.

    def _extend_until(self, done: threading.Event) -> None:
        while not done.wait(1):
            self._extend()

    async def working_dir(self) -> str:
        """The canonical directory commands start in, resolved once with `pwd -P`."""
        if self._resolved_working_dir is None:
            result = await self.run(['pwd', '-P'])
            resolved = result.stdout.removesuffix('\n')
            if result.exit_code != 0 or not posixpath.isabs(resolved):
                raise WorkspaceError(f'Could not resolve the working directory: {result.stderr.strip()}')
            self._resolved_working_dir = resolved
        return self._resolved_working_dir

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """Run a detached Boat command until it exits; a timeout or cancellation stops its whole process tree."""
        validate_timeout(timeout)
        if isinstance(command, str) != shell:
            raise TypeError('pass a string command with shell=True, or an argv sequence with shell=False')
        argv = ['/bin/sh', '-c', command] if isinstance(command, str) else list(command)
        if not argv:
            raise ValueError('command must not be empty')
        sandbox_id = await self._connect()
        api = self._boat()
        line = _command_line(argv, self._working_dir, {**self._env, **(env or {})})
        started = await self._call(partial(api.command, sandbox_id, boat.CommandRequest(command=line, detached=True)))
        if not isinstance(started.actual_instance, boat.CommandStartedResponse):
            raise WorkspaceError('Boat did not start the command detached.')
        pid = started.actual_instance.process_id
        output: dict[str, list[str]] = {'stdout': [], 'stderr': []}

        def follow() -> dict[str, Any]:
            done, sizes = threading.Event(), {'stdout': 0, 'stderr': 0}
            # Boat stops an idle sandbox even while a command runs, so keep extending it until the command ends.
            threading.Thread(target=self._extend_until, args=(done,), daemon=True).start()
            try:
                for frame in _follow(api, sandbox_id, pid):
                    kind = frame.get('type')
                    if kind not in output:
                        return frame  # `exit`, or `error`: the last frame.
                    output[kind].append(data := str(frame.get('data') or ''))
                    sizes[kind] += len(data.encode())
                    if sizes[kind] > _OUTPUT_LIMIT:
                        return {'type': 'limit'}
                return {'type': 'error', 'message': 'the output stream ended early'}
            finally:
                done.set()

        frame: dict[str, Any] | None = None
        try:
            with anyio.move_on_after(timeout):
                frame = await self._call(follow, abandon=True)
        except BaseException:
            await self._stop(sandbox_id, pid)
            raise
        stdout, stderr = ''.join(output['stdout']), ''.join(output['stderr'])
        if frame is None or frame['type'] == 'limit':
            if (gone := await self._stop(sandbox_id, pid)) is not None:
                raise gone
            if frame is None:
                raise WorkspaceTimeoutError(f'Command timed out after {timeout:g}s', stdout=stdout, stderr=stderr)
            message = f'Command output exceeded {_OUTPUT_LIMIT} bytes per stream; redirect it to a file instead'
            raise WorkspaceOutputLimitError(message, limit=_OUTPUT_LIMIT, stdout=stdout, stderr=stderr)
        exit_code = frame.get('exitCode')
        if frame['type'] != 'exit' or not isinstance(exit_code, int):
            # Boat lost the command: find out whether the sandbox was deleted or stopped meanwhile.
            state = (await self._call(partial(api.get, sandbox_id))).sandbox.state
            if state not in ('ready', 'idle', 'running'):
                raise WorkspaceUnavailableError(f'Boat sandbox {sandbox_id!r} is {state}.')
            raise WorkspaceError(f'Boat lost track of the command: {frame.get("message") or frame.get("status")}')
        if exit_code == 125 and stdout == _MISSING_CWD:
            raise WorkspaceError(f'working_dir {self._working_dir!r} does not exist in Boat sandbox {sandbox_id}.')
        return CommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr)

    async def _stop(self, sandbox_id: str, process_id: int) -> WorkspaceUnavailableError | None:
        """Stop a command and every process it started; shielded, bounded, best effort.

        Returns the error to raise instead if the sandbox is gone: a deleted sandbox's output stream can hang.
        """
        stop = partial(self._boat().stop_command, sandbox_id, process_id, _request_timeout=_STOP_TIMEOUT)
        with anyio.move_on_after(_STOP_TIMEOUT, shield=True):
            try:
                await self._call(stop, abandon=True)
            except WorkspaceUnavailableError as error:
                return error
            except Exception:
                pass
        return None

    # Boat's file API moves at most 5 MiB and reaches only paths under /home/user and /tmp. Anything it refuses
    # (a larger file, another path, a path error) goes through the shell, which raises the built-in errors.
    async def read_bytes(self, path: str) -> bytes:
        sandbox_id = await self._connect()
        try:
            file = await self._call(partial(self._boat().read_file, sandbox_id, path, encoding='base64'))
        except boat.ApiException as error:
            if _status(error) != 400:
                raise
            return await self._shell.read_bytes(path)
        return base64.b64decode(file.content)

    async def write_bytes(self, path: str, data: bytes) -> None:
        sandbox_id = await self._connect()
        if len(data) <= _FILE_API_LIMIT:
            request = boat.FileWriteRequest(path=path, content=base64.b64encode(data).decode(), encoding='base64')
            try:
                # Creates missing parents and writes through a symlink.
                await self._call(partial(self._boat().write_file, sandbox_id, request))
                return
            except boat.ApiException as error:
                if _status(error) != 400:
                    raise
        await self._shell.write_bytes(path, data)

    async def stat(self, path: str) -> FileEntry:
        return await self._shell.stat(path)

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        return await self._shell.list_dir(path)

    async def make_dir(self, path: str) -> None:
        await self._shell.make_dir(path)

    async def remove(self, path: str) -> None:
        await self._shell.remove(path)

    async def exists(self, path: str) -> bool:
        return await self._shell.exists(path)

    async def realpath(self, path: str) -> str:
        return await self._shell.realpath(path)


@dataclass(kw_only=True)
class BoatSandbox(AbstractCapability[AgentDepsT]):
    """Supply a [Boat](https://boat.dev) sandbox as the run's workspace.

    A run without a ref creates a sandbox on first use; a `WorkspaceRef` from an earlier run (`workspace=` or
    the message history) attaches to it. Ending a run never deletes the sandbox; `destroy` does. Compose it
    with capabilities that use the workspace, such as Pydantic AI Harness's `Coder`, `Shell` and `FileSystem`.
    """

    api: boat.BoatApi | None = None
    """Boat SDK client to use, which you own; `None` builds one from `BOAT_API_KEY` on first use."""

    machine_type: MachineType | None = None
    """Size of a new sandbox: `'small'`, `'default'`, `'large'` or `'xlarge'`; Boat's `default` when `None`."""

    snapshot: str | None = None
    """Named Boat snapshot, or `bsh_` share code, a new sandbox starts from; Boat's base image when `None`."""

    auto_stop: int | None = None
    """Seconds a new sandbox may go without operations before Boat stops it, keeping its disk; `0` disables
    it and `None` keeps Boat's default of an hour. Operations and running commands extend it."""

    working_dir: str | None = None
    """Absolute directory commands start in and relative paths resolve against; created in a new sandbox."""

    env: Mapping[str, str] | None = field(default=None, repr=False)
    """Environment variables every command gets. They travel in the command line, so Boat's command history
    shows them."""

    def __post_init__(self) -> None:
        if self.defer_loading:
            raise UserError('`BoatSandbox` cannot be deferred: a run picks its workspace when it starts.')
        if self.working_dir is not None and not posixpath.isabs(self.working_dir):
            raise UserError(f'working_dir must be an absolute path or None, got {self.working_dir!r}.')
        if self.auto_stop is not None and (type(self.auto_stop) is not int or self.auto_stop < 0):
            raise UserError(f'auto_stop must be a number of seconds (0 to disable) or None, got {self.auto_stop!r}.')

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> BoatSandboxBackend | None:
        """Build the run's backend without I/O, or return `None` for another provider's ref."""
        if ref is not None and ref.provider != 'boat':
            return None
        return BoatSandboxBackend(
            api=self.api,
            ref=ref,
            machine_type=self.machine_type,
            snapshot=self.snapshot,
            auto_stop=self.auto_stop,
            working_dir=self.working_dir,
            env=self.env,
        )

    async def destroy(self, ref: WorkspaceRef) -> None:
        """Delete the sandbox `ref` names, running or stopped; one already gone returns quietly."""
        if ref.provider != 'boat':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'boat'")
        try:
            await anyio.to_thread.run_sync(
                partial(boat_helpers.stop_and_remove, self.api or _default_api(), ref.id, delete=True)
            )
        except boat.ApiException as error:
            if _status(error) != 404:  # Already gone.
                raise
