"""The public inference pipeline around :mod:`noelle.model`."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from noelle.model import Arch, BackboneFamily, Runtime
from noelle.answers import ChoiceAnswer, NoelleResponse, NoulAnswer, ScoreAnswer
from noelle.questions import Choice, Noul, Question, Score


DecisionKind = Literal["choice", "score", "noul"]
TOKENIZER_ID = "HuggingFaceTB/SmolLM2-135M"
TOKENIZER_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"


@dataclass(frozen=True)
class PreparedInput:
    ids: jax.Array
    ids_mask: jax.Array
    ctx_ids: jax.Array
    ctx_mask: jax.Array
    choice_count: int
    prompt_truncated: bool
    choices_truncated: int


@dataclass(frozen=True)
class Decision:
    options: tuple[str, ...]
    probabilities: tuple[float, ...]
    selected_index: int
    prompt_truncated: bool
    choices_truncated: int

    @property
    def selected_option(self) -> str:
        return self.options[self.selected_index]


class DecisionPipeline:
    """Prepare raw inputs, run the selected architecture, and format its choice probabilities."""

    def __init__(
        self, runtime: Runtime, tokenizer: Tokenizer, *,
        max_prompt_tokens: int = 1024, max_option_tokens: int = 256,
    ):
        if max_prompt_tokens < 2 or max_option_tokens < 1:
            raise ValueError("max_prompt_tokens must be >= 2 and max_option_tokens >= 1")
        self._runtime = runtime
        self._tokenizer = tokenizer
        self.max_prompt_tokens = max_prompt_tokens
        self.max_option_tokens = max_option_tokens

    @property
    def arch(self) -> Arch:
        return self._runtime.arch

    @property
    def backbone_family(self) -> BackboneFamily:
        return self.arch.family

    def preprocess(
        self, *, state: Any, question: str, options: Sequence[str],
        kind: DecisionKind = "choice",
    ) -> PreparedInput:
        """Use the selected architecture's training prompt and token layout."""
        if kind not in ("choice", "score", "noul"):
            raise ValueError("kind must be 'choice', 'score', or 'noul'")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be nonempty text")
        if isinstance(options, (str, bytes)) or not isinstance(options, Sequence):
            raise ValueError("options must be a sequence of at least two strings")
        choices = tuple(options)
        if len(choices) < 2 or any(not isinstance(option, str) or not option.strip() for option in choices):
            raise ValueError("options must contain at least two nonempty strings")

        if self.backbone_family is BackboneFamily.DB:
            return self._preprocess_db(state, question, choices, kind)
        raise NotImplementedError(f"no preprocessor for {self.backbone_family.value} yet")

    def _preprocess_db(
        self, state: Any, question: str, options: tuple[str, ...], kind: DecisionKind,
    ) -> PreparedInput:
        if isinstance(state, str):
            state_text = state
        else:
            try:
                state_text = json.dumps(
                    state, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            except (TypeError, ValueError) as error:
                raise ValueError("state must be text or JSON-serializable") from error
        prompt_text = f"State:\n{state_text}\nTask: {kind}\nQuestion:\n{question}\nAnswer:"
        prompt = [1] + self._tokenizer.encode(prompt_text, add_special_tokens=False).ids
        prompt_truncated = len(prompt) > self.max_prompt_tokens
        if prompt_truncated:
            prompt = [prompt[0]] + prompt[-(self.max_prompt_tokens - 1):]

        choices = [self._tokenizer.encode(option, add_special_tokens=False).ids for option in options]
        if any(not tokens for tokens in choices):
            raise ValueError("each option must contain at least one token")
        choices_truncated = sum(len(tokens) > self.max_option_tokens for tokens in choices)
        choices = [tokens[:self.max_option_tokens] for tokens in choices]

        # Match train.collate's fixed token widths and power-of-two choice slots.
        choice_slots = 1 << (len(choices) - 1).bit_length()
        ids = np.zeros((1, self.max_prompt_tokens), dtype=np.int32)
        ids_mask = np.zeros_like(ids, dtype=bool)
        ctx_ids = np.zeros((1, choice_slots, self.max_option_tokens), dtype=np.int32)
        ctx_mask = np.zeros_like(ctx_ids, dtype=bool)
        ids[0, -len(prompt):] = prompt
        ids_mask[0, -len(prompt):] = True
        for index, tokens in enumerate(choices):
            ctx_ids[0, index, :len(tokens)] = tokens
            ctx_mask[0, index, :len(tokens)] = True
        return PreparedInput(
            jnp.asarray(ids), jnp.asarray(ids_mask), jnp.asarray(ctx_ids),
            jnp.asarray(ctx_mask), len(options), prompt_truncated, choices_truncated,
        )

    def predict(
        self, *, state: Any, question: str, options: Sequence[str],
        kind: DecisionKind = "choice",
    ) -> Decision:
        prepared = self.preprocess(state=state, question=question, options=options, kind=kind)
        choices = tuple(options)
        option_lengths = np.count_nonzero(
            np.asarray(jax.device_get(prepared.ctx_mask))[0, :prepared.choice_count], axis=-1,
        )
        if np.all(option_lengths == 1):
            warnings.warn(
                "Every option is one token. This checkpoint's cross-attention then "
                "ignores the state and question; the probabilities are option priors. "
                "Use descriptive options, and retrain with a query residual for reliable Noul predictions.",
                UserWarning, stacklevel=2,
            )
        logits = np.asarray(jax.device_get(self._runtime.logits(
            prepared.ids, prepared.ids_mask, prepared.ctx_ids, prepared.ctx_mask,
            choice_count=prepared.choice_count,
        )), dtype=np.float64)
        if logits.shape != (len(choices),) or not np.isfinite(logits).all():
            raise ValueError("model returned invalid choice logits")
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        return Decision(
            choices, tuple(float(value) for value in probabilities),
            int(probabilities.argmax()), prepared.prompt_truncated,
            prepared.choices_truncated,
        )

    __call__ = predict


def available_models() -> tuple[Arch, ...]:
    return tuple(Arch)


def load_pipeline(
    model: Arch = Arch.SmollmDB135m, *, checkpoint: str | Path | None = None,
    tokenizer: str | Path = TOKENIZER_ID, max_prompt_tokens: int = 1024,
    max_option_tokens: int = 256,
) -> DecisionPipeline:
    """Choose an architecture and return the outer preprocessing pipeline."""
    if not isinstance(model, Arch):
        raise TypeError(f"model must be an Arch value, got {model!r}; available: {available_models()}")
    tokenizer_path = Path(tokenizer)
    if not tokenizer_path.is_file():
        tokenizer_path = Path(hf_hub_download(
            repo_id=str(tokenizer), filename="tokenizer.json", revision=TOKENIZER_REVISION,
        ))
    tokenizer_instance = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer_instance.get_vocab_size() != 49152:
        raise ValueError("tokenizer vocabulary does not match the checkpoint")
    return DecisionPipeline(
        Runtime(model, checkpoint=checkpoint), tokenizer_instance,
        max_prompt_tokens=max_prompt_tokens, max_option_tokens=max_option_tokens,
    )


def _render_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("instructions and criteria must be text or JSON-serializable") from error


class NoellePipeline:
    """Answer a named set of Choice, Noul, and Score questions about one state."""

    def __init__(
        self, arch: Arch = Arch.SmollmDB135m, *, checkpoint: str | Path | None = None,
        tokenizer: str | Path = TOKENIZER_ID, max_prompt_tokens: int = 1024,
        max_option_tokens: int = 256,
    ):
        self._decision = load_pipeline(
            arch, checkpoint=checkpoint, tokenizer=tokenizer,
            max_prompt_tokens=max_prompt_tokens,
            max_option_tokens=max_option_tokens,
        )

    @property
    def arch(self) -> Arch:
        return self._decision.arch

    def __call__(self, *, state: Any, questions: Mapping[str, Question]) -> NoelleResponse:
        if not isinstance(questions, Mapping) or not questions:
            raise ValueError("questions must be a nonempty mapping")
        answers = {}
        for name, question in questions.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("each question ID must be nonempty text")
            if not isinstance(question, (Choice, Noul, Score)):
                raise TypeError(f"question {name!r} must be Choice, Noul, or Score")
            if question.instructions is None:
                raise ValueError(f"question {name!r} needs instructions")
            instructions = _render_content(question.instructions)
            if not instructions.strip():
                raise ValueError(f"question {name!r} needs nonempty instructions")

            if isinstance(question, Choice):
                if not isinstance(question.criteria, Mapping) or len(question.criteria) < 2:
                    raise ValueError(f"Choice {name!r} needs at least two named criteria")
                labels = tuple(question.criteria)
                if any(not isinstance(label, str) or not label.strip() for label in labels):
                    raise ValueError(f"Choice {name!r} has an empty or non-text label")
                options = [
                    label if description is None else f"{label}: {_render_content(description)}"
                    for label, description in question.criteria.items()
                ]
                decision = self._decision.predict(
                    state=state, question=instructions, options=options, kind="choice",
                )
                answers[name] = ChoiceAnswer(
                    labels[decision.selected_index],
                    dict(zip(labels, decision.probabilities, strict=True)),
                )
            elif isinstance(question, Noul):
                criteria = question.criteria or {}
                if not isinstance(criteria, Mapping) or set(criteria) - {"false", "true"}:
                    raise ValueError(f"Noul {name!r} criteria may only have 'false' and 'true'")
                options = [
                    label if criteria.get(key) is None else f"{label}: {_render_content(criteria[key])}"
                    for key, label in (
                        ("true", "agree with the instruction"),
                        ("false", "disagree with the instruction"),
                    )
                ]
                decision = self._decision.predict(
                    state=state, question=instructions, options=options, kind="noul",
                )
                answers[name] = NoulAnswer(decision.probabilities[0])
            else:
                if isinstance(question.criteria, (str, bytes)) or not isinstance(question.criteria, Sequence) or len(question.criteria) < 2:
                    raise ValueError(f"Score {name!r} needs at least two ordered levels")
                options = [_render_content(level) for level in question.criteria]
                decision = self._decision.predict(
                    state=state, question=instructions, options=options, kind="score",
                )
                probabilities = dict(enumerate(decision.probabilities))
                answers[name] = ScoreAnswer(
                    sum(index * probability for index, probability in probabilities.items()),
                    probabilities, dict(enumerate(question.criteria)),
                )
        return NoelleResponse(answers)
