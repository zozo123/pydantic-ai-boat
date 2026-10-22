"""Tests against the real Boat API; they skip without `BOAT_API_KEY`.

Every sandbox they start is `small`, auto-stops after 15 minutes if a test is killed before its cleanup, and
is deleted at the end.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Awaitable, Callable

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef, WorkspaceTimeoutError
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.shell import Shell

from pydantic_ai_boat import BoatSandbox, BoatSandboxBackend

pytestmark = [
    pytest.mark.skipif(not os.environ.get('BOAT_API_KEY'), reason='set BOAT_API_KEY to test against Boat'),
    pytest.mark.anyio,
]

AUTO_STOP = 900


@pytest.fixture(scope='module')
def anyio_backend() -> str:
    return 'asyncio'


def new_backend() -> BoatSandboxBackend:
    return BoatSandboxBackend(machine_type='small', auto_stop=AUTO_STOP)


async def delete(backend: WorkspaceBackend) -> None:
    if backend.ref is not None:
        await BoatSandbox().destroy(backend.ref)


class TestConformance(WorkspaceBackendSuite):
    """Core's workspace backend rules, against one shared sandbox plus one per destructive rule."""

    @pytest.fixture(scope='class')
    @classmethod
    async def backend(cls) -> AsyncIterator[BoatSandboxBackend]:
        backend = new_backend()
        try:
            yield backend
        finally:
            await delete(backend)

    @pytest.fixture
    async def fresh_backend(self) -> AsyncIterator[Callable[[], WorkspaceBackend]]:
        created: list[BoatSandboxBackend] = []

        def create() -> WorkspaceBackend:
            created.append(new_backend())
            return created[-1]

        yield create
        for backend in created:
            await delete(backend)

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        # Attaching starts nothing; the sandbox belongs to the backend that created it.
        return lambda ref: BoatSandboxBackend(ref=ref)

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return delete

    @pytest.fixture
    def filesystem_honors_shell_permissions(self) -> bool:
        # Boat's file API reads as root, so it reads a file the command user cannot.
        return False


async def test_an_agent_works_in_a_sandbox_and_a_later_run_continues_there() -> None:
    created: list[WorkspaceBackend] = []

    class Tracked(BoatSandbox[None]):
        def get_workspace(self, ctx: RunContext[None], *, ref: WorkspaceRef | None) -> BoatSandboxBackend | None:
            backend = super().get_workspace(ctx, ref=ref)
            if backend is not None:
                created.append(backend)
            return backend

    def run_each_prompt(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """Run the prompt as one `run_command` call, then answer with what it printed."""
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        for part in request.parts:
            if isinstance(part, ToolReturnPart):
                return ModelResponse(parts=[TextPart(str(part.content))])
        prompt = next(part.content for part in request.parts if isinstance(part, UserPromptPart))
        return ModelResponse(parts=[ToolCallPart('run_command', {'command': prompt})])

    sandbox = Tracked(machine_type='small', auto_stop=AUTO_STOP)
    agent = Agent(FunctionModel(run_each_prompt), deps_type=type(None), capabilities=[sandbox, Shell[None]()])
    try:
        first = await agent.run('echo continued > ~/note.txt && uname -s')
        ref = first.workspace.ref
        assert ref is not None and ref.provider == 'boat'
        assert 'Linux' in first.output

        later = await agent.run('cat ~/note.txt', workspace=ref)
        assert 'continued' in later.output
        assert later.workspace.ref == ref
    finally:
        for backend in created:
            await delete(backend)


@pytest.mark.parametrize('anyio_backend', ['trio'])
async def test_runs_on_trio(anyio_backend: str) -> None:
    backend = new_backend()
    try:
        assert (await backend.run(['echo', 'trio'])).stdout == 'trio\n'
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run(['sleep', '60'], timeout=1)
    finally:
        await delete(backend)
