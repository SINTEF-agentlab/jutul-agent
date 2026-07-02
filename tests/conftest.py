"""Test-time configuration.

Loads a project-local ``.env`` so credential-gated integration tests can
discover provider keys before pytest evaluates ``skipif`` conditions, and
resets the module-level workspace / state-home overrides between tests so
state from one test cannot leak into the next.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

from jutul_agent.lab.fakes import FakeJulia, make_fake_adapter
from jutul_agent.paths import set_state_home, set_workspace_root
from jutul_agent.session import Session
from jutul_agent.simulators.base import SimulatorAdapter

load_dotenv()

# Integration tests run real Julia and pay first-call compilation, which can take
# minutes (more on slow CI). The global 120 s timeout would flake them, so give them
# a large budget that only catches a genuine (infinite) deadlock. An explicit
# per-test ``@pytest.mark.timeout`` still wins.
_INTEGRATION_TIMEOUT_S = 1800


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.get_closest_marker("integration") and item.get_closest_marker("timeout") is None:
            item.add_marker(pytest.mark.timeout(_INTEGRATION_TIMEOUT_S))


@pytest.fixture(autouse=True)
def _reset_workspace_overrides(tmp_path: Path, monkeypatch):
    from jutul_agent import paths

    # Use tmp_path as workspace so session output dirs stay isolated and
    # don't pollute the project root.  Tests that need a specific workspace
    # layout use the ``workspace`` fixture which overrides these values.
    paths.set_workspace_root(tmp_path)
    paths.set_state_home(tmp_path / "state")
    # Suppress OS file-open calls (image viewer, browser) during tests.
    monkeypatch.setenv("JUTUL_AGENT_NO_OPEN", "1")
    yield
    paths.set_workspace_root(None)
    paths.set_state_home(None)


@pytest.fixture(autouse=True)
def _offline_titling(monkeypatch):
    """Session titling fires after first turns; unit tests must stay off the network.

    The real function already degrades to ``None`` on failure, but attempting a
    provider call with a placeholder key is slow and unhermetic. Tests of the
    titling flow monkeypatch their own replacement over this one.
    """

    async def _no_title(model_id, conversation):
        return None

    monkeypatch.setattr("jutul_agent.agent.titling.generate_session_title", _no_title)


@pytest.fixture
def fake_julia() -> FakeJulia:
    return FakeJulia()


@pytest.fixture
def fake_adapter(tmp_path: Path) -> SimulatorAdapter:
    return make_fake_adapter(tmp_path)


@pytest.fixture
def session(tmp_path: Path, fake_julia: FakeJulia, fake_adapter: SimulatorAdapter) -> Session:
    return Session.create(julia=fake_julia, state_root=tmp_path, simulator=fake_adapter)


@pytest.fixture
def session_with_pkg(tmp_path: Path) -> tuple[Session, FakeJulia]:
    pkg_root = tmp_path / "FakePkg"
    (pkg_root / "examples").mkdir(parents=True)
    (pkg_root / "examples" / "intro.jl").write_text(
        "using FakePkg\n# canary string\n", encoding="utf-8"
    )
    julia = FakeJulia(
        pkgdir={"FakePkg": pkg_root},
        answers={"2 + 2": "4"},
    )
    adapter = make_fake_adapter(tmp_path)
    session = Session.create(julia=julia, state_root=tmp_path, simulator=adapter)
    return session, julia


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Temporary workspace with an isolated state-home directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    state_home = tmp_path / "state"
    state_home.mkdir()
    set_workspace_root(ws)
    set_state_home(state_home)
    return ws
