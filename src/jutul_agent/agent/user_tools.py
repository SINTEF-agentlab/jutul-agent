import importlib.util
import sys
from pathlib import Path

from jutul_agent.paths import user_simulators_dir, user_tools_dir
from jutul_agent.session import Session


def _load_tool_factories(tools_dir: Path) -> list:
    factories = []
    if not tools_dir.is_dir():
        return factories
    for tool_file in sorted(tools_dir.glob("*.py")):
        spec = importlib.util.spec_from_file_location(tool_file.stem, tool_file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[tool_file.stem] = module
            spec.loader.exec_module(module)
            tool_func_name = f"make_{tool_file.stem}_tool"
            if hasattr(module, tool_func_name):
                factories.append(getattr(module, tool_func_name))
    return factories


def load_user_tools(session: Session):
    # Global tools, plus tools scoped to the active simulator.
    factories = _load_tool_factories(user_tools_dir())
    factories += _load_tool_factories(user_simulators_dir() / session.simulator.name / "tools")
    return [factory(session) for factory in factories]