"""Canaries over the private upstream surface jutul-agent deliberately touches.

A failure here means a deepagents/langgraph bump moved something we compose
with. Runtime degrades gracefully in every case (that is the point of the
guards); these tests make a pin bump fail loudly at test time instead, with a
map of what to fix.
"""

from __future__ import annotations


def test_summarization_engine_private_methods_still_exist() -> None:
    from deepagents.middleware.summarization import SummarizationMiddleware

    from jutul_agent.agent.summarization import _ENGINE_METHODS, manual_compaction_available

    missing = [name for name in _ENGINE_METHODS if not hasattr(SummarizationMiddleware, name)]
    assert manual_compaction_available(), (
        f"deepagents no longer exposes {missing}; manual /compact degrades to "
        "'unavailable' at runtime. Recompose agent/summarization.py against the "
        "new engine."
    )


def test_langgraph_tool_call_writer_still_exposed() -> None:
    from jutul_agent.agent import tools

    assert tools._tool_call_writer is not None, (
        "langgraph moved pregel._tools._tool_call_writer; live run_julia output "
        "streaming silently disappears. Update the import guard in agent/tools.py."
    )


def test_deepagents_validate_path_still_patchable() -> None:
    import deepagents.backends.utils as backend_utils

    assert callable(getattr(backend_utils, "validate_path", None)), (
        "deepagents moved backends.utils.validate_path; the Windows drive-path "
        "shim in agent/windows_paths.py silently no-ops. Re-point the patch."
    )


def test_deepagents_harness_profile_resolver_still_exists() -> None:
    from deepagents.profiles.harness import harness_profiles

    assert hasattr(harness_profiles, "_harness_profile_for_model"), (
        "deepagents moved _harness_profile_for_model; tests/test_builder.py "
        "asserts profile resolution through it."
    )
