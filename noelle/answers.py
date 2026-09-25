"""Results returned by :class:`noelle.NoellePipeline`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]


@dataclass(frozen=True)
class NoulAnswer:
    noul: float


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    probabilities: dict[int, float]
    legend: dict[int, Any]


Answer: TypeAlias = ChoiceAnswer | NoulAnswer | ScoreAnswer


@dataclass(frozen=True)
class NoelleResponse:
    answers: dict[str, Answer]

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        return {key: value for key, value in self.answers.items() if isinstance(value, ChoiceAnswer)}

    @property
    def nouls(self) -> dict[str, NoulAnswer]:
        return {key: value for key, value in self.answers.items() if isinstance(value, NoulAnswer)}

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        return {key: value for key, value in self.answers.items() if isinstance(value, ScoreAnswer)}
