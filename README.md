# pydantic-ai-boat

Run a [Pydantic AI](https://pydantic.dev/docs/ai/) agent's commands and file edits in a [Boat](https://boat.dev) cloud sandbox instead of on your machine. Community package (v0.1.0, installed from Git); not part of Pydantic AI.

It is one module on top of Boat's official Python SDK, [`boat-sdk`](https://pypi.org/project/boat-sdk/). `BoatSandbox` is a Pydantic AI capability that supplies a Boat sandbox, a root Ubuntu VM, as the run's [workspace](https://pydantic.dev/docs/ai/core-concepts/workspace/), so workspace tools such as Pydantic AI Harness's `Coder`, `Shell` and `FileSystem` run there.

## Install

```bash
uv add git+https://github.com/zozo123/pydantic-ai-boat
# or
pip install git+https://github.com/zozo123/pydantic-ai-boat
```

It needs Python 3.11 or newer and depends on `pydantic-ai-slim`, `boat-sdk` and `anyio`. Set `BOAT_API_KEY` to a [Boat API key](https://docs.boat.dev/api-keys). The example below also uses `pydantic-ai-harness` and an Anthropic model.

## Quick start

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai_boat import BoatSandbox
from pydantic_ai_harness.coder import Coder

agent = Agent('anthropic:claude-opus-5-5', capabilities=[BoatSandbox(), Coder()])


async def main() -> None:
    result = await agent.run('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
    ref = result.workspace.ref  # Store it to come back later, from any process with BOAT_API_KEY.

    later = await agent.run('Which capability would you add next?', workspace=ref)
    print(later.output)

    await BoatSandbox().destroy(ref)


asyncio.run(main())
```

A run with no ref creates a sandbox on first use. A ref, passed as `workspace=` or found in the message history, attaches to that sandbox and resumes it if Boat stopped it; `workspace='new'` starts a fresh one. Attaching to a deleted sandbox raises `WorkspaceUnavailableError` instead of creating a replacement.

## Options

| Option | Default | What it does |
| --- | --- | --- |
| `api` | `None` | A `boat_sdk.BoatApi` client you own. `None` builds one from `BOAT_API_KEY` on first use. |
| `machine_type` | `None` (Boat's `default`) | Size of a new sandbox: `'small'`, `'default'`, `'large'` or `'xlarge'`. |
| `snapshot` | `None` (Boat's base image) | Named Boat snapshot, or `bsh_` share code, a new sandbox starts from. |
| `auto_stop` | `None` (1 hour) | Seconds without activity before Boat stops a new sandbox, keeping its disk. `0` disables it. |
| `working_dir` | `None` (`/home/user`) | Absolute directory commands start in and relative paths resolve against. Created in a new sandbox. |
| `env` | `None` | Environment variables every command gets. Nothing is read from your machine. |

`BoatSandboxBackend` takes the same options plus `ref`. Build it yourself to prepare a sandbox before a run, for example `await Workspace(backend).write_text('TASK.md', ...)`, then pass `workspace=backend`.

## Behavior

- **No account secrets.** A new sandbox is created with Boat's `noEnv`. Without it, Boat gives every command your GitHub token and Boat tokens, which code the agent runs could read and use. Pass what a task needs with `env`.
- **`env` values are visible in Boat's command history.** Boat's command API has no environment field, so the values travel in each command line (`env -- NAME=value ...`), where the command history and the sandbox's process list show them.
- **Auto-stop is extended while you work.** Boat counts `auto_stop` from create or resume, and commands do not move it. Every operation, and every running command, pushes the deadline to a full `auto_stop` from now, at most once a minute (or once per half of `auto_stop` when that is shorter). An idle sandbox stops; a later run that attaches resumes it. Boat keeps a stopped sandbox's disk for 14 days.
- **Commands** run as Boat's `user` (passwordless `sudo`) with stdin at EOF; a shell string runs under `/bin/sh -c`. Each runs as a detached Boat command, so there is no 600-second cap. A timeout or a cancelled run stops the command's whole process tree. Output over 8 MiB per stream raises `WorkspaceOutputLimitError`.
- **Files.** Reads and writes of up to 5 MiB under `/home/user` and `/tmp`, the paths Boat's file API reaches, use that API. Larger files, other paths, and path errors fall back to shell commands: slower, 64 KiB per command, but they reach the whole VM and raise the built-in errors such as `FileNotFoundError` and `IsADirectoryError`. Boat's file API reads as root, so a read can return a file the command user cannot open.
- **`destroy(ref)`** deletes a sandbox, running or stopped, and returns quietly if it is already gone. Nothing else deletes it: ending a run leaves it running until auto-stop.
- **asyncio and Trio.** The SDK is synchronous, so its calls run in worker threads and there is no client to close.

## Running the tests

There is no fake server: `tests/test_live.py` runs only against the real Boat API and skips without `BOAT_API_KEY`.

```bash
uv sync
uv run ruff check && uv run ruff format --check && uv run pyright
BOAT_API_KEY=... uv run pytest tests -q
```

The tests run Pydantic AI's workspace backend conformance suite (`WorkspaceBackendSuite`) against Boat, an agent run that continues in the same sandbox through `workspace=result.workspace.ref`, and a check on Trio. A run starts 6 `small` sandboxes, deletes each one (each also auto-stops after 15 minutes in case a run is killed), and takes about 3 minutes. Boat limits sandbox starts per minute, hour and day by plan. CI runs them when the repository has a `BOAT_API_KEY` secret.

## Status

A community package, not part of Pydantic AI. A first-party integration was proposed in [pydantic/pydantic-ai#10967](https://github.com/pydantic/pydantic-ai/pull/10967) and closed in favour of shipping it as this third-party package. While it is on 0.x, the API may change between minor releases.

## License

MIT; see [LICENSE](LICENSE).
