from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Callable, List, Protocol

from ..config import PreferenceConfig, UpdateConfig
from ..extractors.preference_extractor import ExtractionState, PreferenceExtractor
from ..llms.models import LLM
from .prompting import build_prompt
from .types import PreferenceFusion
from .update import PAMTState, PAMTUpdater


History = List[str]
logger = logging.getLogger(__name__)


class Baseline(Protocol):
    def generate(self, prompt: str) -> str:
        ...


PromptBuilder = Callable[[History, str], str]


def default_prompt_builder(history: History, user_text: str) -> str:
    if not history:
        return user_text
    joined = "\n".join(history)
    return f"{joined}\n\nUser: {user_text}"


@dataclass
class PlainBaseline:
    llm: LLM
    prompt_builder: PromptBuilder = default_prompt_builder

    def generate(self, prompt: str) -> str:
        return self.llm.generate(prompt)


@dataclass
class PAMTAgent:
    llm: LLM
    extractor: PreferenceExtractor
    pref_config: PreferenceConfig
    update_config: UpdateConfig
    updater: PAMTUpdater = field(init=False)
    history: History = field(default_factory=list)
    extractor_state: ExtractionState = field(default_factory=ExtractionState)
    update_state: PAMTState = field(default_factory=PAMTState)

    def respond(
        self,
        user_text: str,
        assistant_text: str | None = None,
        assistant_token_count: int | None = None,
        base_prompt: str | None = None,
    ) -> tuple[str, PreferenceFusion]:
        # Use the latest assistant response to update preference signals.
        if assistant_text is None:
            assistant_text = ""
        pref = self.extractor.extract(
            user_text,
            assistant_text,
            self.extractor_state,
            assistant_token_count,
        )
        fusion, change = self.updater.update(self.update_state, pref)
        logger.debug(
            "PAMTAgent.respond: pref(len=%.3f dens=%.3f form=%.3f) change=%s",
            pref.length,
            pref.density,
            pref.formality,
            change.overall_triggered,
        )
        # If a baseline prompt is provided, inject PAMT control into it.
        prompt_source = base_prompt if base_prompt is not None else user_text
        prompt = build_prompt(prompt_source, fusion, self.pref_config)
        response = self.llm.generate(prompt)
        self.history.append(f"User: {user_text}")
        self.history.append(f"Assistant: {response}")
        return response, fusion

    def __post_init__(self) -> None:
        self.updater = PAMTUpdater(self.update_config)




