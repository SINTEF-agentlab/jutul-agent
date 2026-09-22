"""Modal dialogs for selecting a model, entering an API key, and pulling a model.

``ModelMenu`` lists discovered models grouped by provider; plus a "Recent"
section; and accepts any ``provider:model`` typed in; it returns
``(model_id, scope)``. ``ApiKeyModal`` collects a provider key when one is
missing. ``OllamaPullModal`` pulls a local model. The caller (``app.TUIApp``)
applies the resulting switch.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import ClassVar

from rich.text import Text
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, ProgressBar, Static
from textual.widgets.option_list import Option

from jutul_agent.models import (
    OLLAMA_CLOUD,
    PROVIDERS,
    RECOMMENDED_OLLAMA_LOCAL,
    ModelInfo,
    discover_available_models,
    discover_models,
    provider_of,
)
from jutul_agent.ollama_client import PullProgress
from jutul_agent.recent_models import load_recent_models

ModelChoice = tuple[str, str]

# Recent rows duplicate a model that also appears in its provider section, so
# their OptionList ids are namespaced to stay unique; ``_model_id`` undoes it.
_RECENT_PREFIX = "recent::"


def _model_id(option_id: str | None) -> str | None:
    if option_id is None:
        return None
    return option_id.removeprefix(_RECENT_PREFIX)


def _matches(model: ModelInfo, needle: str) -> bool:
    if not needle:
        return True
    return (
        needle in model.id.lower() or needle in model.label.lower() or needle in model.note.lower()
    )


def _render_model(model: ModelInfo, *, current: str | None, full_id: bool = False) -> Text:
    """One catalog row. ``full_id`` shows ``provider:model`` (for cross-provider
    rows like Recent); otherwise just the model name, since the provider header
    already gives the context."""
    text = Text()
    marker = "● " if model.id == current else "  "
    text.append(marker, style="cyan" if model.id == current else "")
    text.append(model.id if full_id else model.label)
    if model.note:
        text.append(f"  — {model.note}", style="dim italic")
    return text


class ModelMenu(ModalScreen[ModelChoice | None]):
    """Filterable model picker. Dismisses with ``(model_id, scope)`` or ``None``."""

    DEFAULT_CSS = """
    ModelMenu {
        align: center middle;
    }
    #model-menu {
        width: 84;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        border: round $surface-lighten-2;
        background: $surface;
        padding: 1 2;
    }
    #model-menu-title {
        text-style: bold;
        padding-bottom: 1;
    }
    /* A clean round border; the Input default is a `tall` half-block border
       that leaves stray edge marks inside the modal. */
    #model-filter {
        border: round $surface-lighten-2;
    }
    #model-filter:focus {
        border: round $primary;
    }
    #model-options {
        height: auto;
        max-height: 16;
        margin: 1 0;
        background: $surface;
    }
    #model-menu-hint {
        color: $text-muted;
        padding-top: 1;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
        # Confirm + set as the global default; plain Enter saves to the workspace.
        Binding("ctrl+a", "select_global", "All workspaces", show=False, priority=True),
    ]

    def __init__(self, *, current: str | None = None) -> None:
        super().__init__()
        self._current = current
        self._recent: list[str] = load_recent_models()
        self._catalog = discover_models()
        # Ollama ships no static profiles, so its models are discovered from the
        # daemon by a worker after mount; empty until (and unless) it answers.
        self._ollama: list[ModelInfo] = []

    def compose(self):
        with Vertical(id="model-menu"):
            yield Static("Select a model", id="model-menu-title")
            yield Input(
                placeholder="Filter, or type any provider:model and press Enter",
                id="model-filter",
            )
            yield OptionList(id="model-options")
            yield Static(
                "Enter · this workspace\nCtrl+A · all workspaces · Esc cancel",
                id="model-menu-hint",
            )

    def on_mount(self) -> None:
        self._populate("")
        self.query_one("#model-filter", Input).focus()
        self.run_worker(self._discover_ollama())
        self.run_worker(self._discover_cloud())

    async def _discover_cloud(self) -> None:
        """Refresh API-listed models without holding up the picker."""
        import asyncio

        self._catalog = await asyncio.to_thread(discover_available_models)
        if self.is_mounted:
            self._populate(self.query_one("#model-filter", Input).value)

    async def _discover_ollama(self) -> None:
        """Merge locally-installed Ollama models into the catalog, best-effort."""
        from jutul_agent import ollama_client

        try:
            names = await ollama_client.installed_models()
        except Exception:  # daemon down or unreachable; leave Ollama out
            return
        if not names:
            return
        self._ollama = [ModelInfo(f"ollama:{name}", name) for name in names]
        if self.is_mounted:
            self._populate(self.query_one("#model-filter", Input).value)

    def _sections(self) -> list[tuple[str, list[ModelInfo]]]:
        """``(header, models)`` sections in display order: Recent, cloud providers,
        then Ollama local (installed + recommended-to-pull) and Ollama cloud."""
        sections: list[tuple[str, list[ModelInfo]]] = []
        if self._recent:
            sections.append(("Recent", [ModelInfo(mid, mid) for mid in self._recent]))
        catalog = self._catalog
        for provider, info in PROVIDERS.items():
            if info.local:
                continue  # Ollama handled below as local + cloud
            if models := list(catalog.get(provider, ())):
                sections.append((info.label, models))
        if local := self._ollama_local_models():
            sections.append(("Ollama (local)", local))
        sections.append(
            ("Ollama (cloud)", [ModelInfo(f"ollama:{tag}", tag) for tag in OLLAMA_CLOUD])
        )
        return sections

    def _ollama_local_models(self) -> list[ModelInfo]:
        """Installed local models (daemon-discovered) plus recommended ones not yet
        pulled, marked so Enter triggers an in-app pull."""
        models = list(self._ollama)
        installed_bases = {m.label.split(":", 1)[0] for m in self._ollama}
        for tag in RECOMMENDED_OLLAMA_LOCAL:
            if tag.split(":", 1)[0] not in installed_bases:
                models.append(ModelInfo(f"ollama:{tag}", tag, "not pulled · Enter to pull"))
        return models

    def _populate(self, filter_text: str) -> None:
        options = self.query_one("#model-options", OptionList)
        options.clear_options()
        raw = filter_text.strip()
        needle = raw.lower()

        # Free-text row: any provider:model not already in the catalog.
        if ":" in raw and not any(
            model.id == raw for models in self._catalog.values() for model in models
        ):
            options.add_option(Option(Text.assemble(("Use ", "bold"), (raw, "bold cyan")), id=raw))

        first_enabled: int | None = None
        index = 0
        for header, models in self._sections():
            visible = [model for model in models if _matches(model, needle)]
            if not visible:
                continue
            is_recent = header == "Recent"
            options.add_option(Option(Text(header, style="bold"), disabled=True))
            index += 1
            for model in visible:
                row = _render_model(model, current=self._current, full_id=is_recent)
                option_id = f"{_RECENT_PREFIX}{model.id}" if is_recent else model.id
                options.add_option(Option(row, id=option_id))
                if first_enabled is None:
                    first_enabled = index
                index += 1

        if first_enabled is not None:
            options.highlighted = first_enabled

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "model-filter":
            self._populate(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._confirm("workspace")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        model_id = _model_id(event.option.id)
        if model_id:
            self._select(model_id, "workspace")

    def on_key(self, event) -> None:
        # While the filter input holds focus, ↑/↓ still drive the option list.
        if event.key in ("down", "up") and isinstance(self.focused, Input):
            options = self.query_one("#model-options", OptionList)
            if event.key == "down":
                options.action_cursor_down()
            else:
                options.action_cursor_up()
            event.stop()

    def action_select_global(self) -> None:
        self._confirm("global")

    def _confirm(self, scope: str) -> None:
        target = self._current_target()
        if target is not None:
            self._select(target, scope)

    def _current_target(self) -> str | None:
        """What Enter / Ctrl+A would pick: a typed provider:model, else the highlight."""
        raw = self.query_one("#model-filter", Input).value.strip()
        if ":" in raw:
            return raw
        options = self.query_one("#model-options", OptionList)
        if options.highlighted is None:
            return None
        return _model_id(options.get_option_at_index(options.highlighted).id)

    def _select(self, model_id: str, scope: str) -> None:
        self.dismiss((model_id, scope))

    def action_cancel(self) -> None:
        self.dismiss(None)


class ApiKeyModal(ModalScreen[str | None]):
    """Masked prompt for a provider API key. Dismisses with the key or ``None``."""

    DEFAULT_CSS = """
    ApiKeyModal {
        align: center middle;
    }
    #api-key-modal {
        width: 72;
        max-width: 90%;
        height: auto;
        border: round $surface-lighten-2;
        background: $surface;
        padding: 1 2;
    }
    #api-key-title {
        text-style: bold;
    }
    #api-key-desc {
        color: $text-muted;
        padding: 1 0;
    }
    #api-key-hint {
        color: $text-muted;
        padding-top: 1;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    def __init__(self, *, env_var: str, provider_label: str) -> None:
        super().__init__()
        self._env_var = env_var
        self._provider_label = provider_label

    def compose(self):
        with Vertical(id="api-key-modal"):
            yield Static(f"{self._provider_label} needs an API key", id="api-key-title")
            yield Static(
                f"Enter {self._env_var}. It's saved for future runs and used right away.",
                id="api-key-desc",
            )
            yield Input(placeholder=self._env_var, id="api-key-input")
            yield Static("Enter to save · Esc to cancel", id="api-key-hint")

    def on_mount(self) -> None:
        self.query_one("#api-key-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def action_cancel(self) -> None:
        self.dismiss(None)


Puller = Callable[[str], AsyncIterator[PullProgress]]


class OllamaPullModal(ModalScreen[bool]):
    """Pull a local model. Dismisses ``True`` on success; Esc cancels or closes."""

    DEFAULT_CSS = """
    OllamaPullModal {
        align: center middle;
    }
    #ollama-pull {
        width: 72;
        max-width: 90%;
        height: auto;
        border: round $surface-lighten-2;
        background: $surface;
        padding: 1 2;
    }
    #ollama-pull-title {
        text-style: bold;
        padding-bottom: 1;
    }
    #ollama-pull-status {
        color: $text-muted;
        padding-top: 1;
    }
    #ollama-pull-hint {
        color: $text-muted;
        padding-top: 1;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    def __init__(self, *, model_name: str, puller: Puller | None = None) -> None:
        super().__init__()
        self._model_name = model_name
        self._puller = puller
        self._worker = None

    def compose(self):
        with Vertical(id="ollama-pull"):
            yield Static(f"Pulling {self._model_name} from Ollama…", id="ollama-pull-title")
            yield ProgressBar(total=100, show_eta=False, id="ollama-pull-bar")
            yield Static("Starting…", id="ollama-pull-status")
            yield Static(
                "This can take a while for large models · Esc to cancel", id="ollama-pull-hint"
            )

    def on_mount(self) -> None:
        self._worker = self.run_worker(self._run(), exclusive=True)

    async def _run(self) -> None:
        from jutul_agent import ollama_client

        puller = self._puller or ollama_client.pull
        try:
            async for progress in puller(self._model_name):
                self._render_progress(progress)
            self.dismiss(True)
        except Exception as exc:  # connection refused, model not found, ...
            self.query_one("#ollama-pull-status", Static).update(f"Pull failed: {exc}")
            self.query_one("#ollama-pull-hint", Static).update("Esc to close")

    def _render_progress(self, progress: PullProgress) -> None:
        bar = self.query_one("#ollama-pull-bar", ProgressBar)
        status = self.query_one("#ollama-pull-status", Static)
        label = progress.status or "working"
        if progress.fraction is not None:
            bar.update(total=100, progress=progress.fraction * 100)
            label = f"{label} · {progress.fraction * 100:.0f}%"
        status.update(label)

    def action_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self.dismiss(False)


__all__ = [
    "ApiKeyModal",
    "ModelChoice",
    "ModelMenu",
    "OllamaPullModal",
    "provider_of",
]
