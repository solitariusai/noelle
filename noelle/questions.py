"""Question declarations accepted by :class:`noelle.NoellePipeline`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TypeAlias


@dataclass(frozen=True)
class Choice:
    """Choose one named criterion; values describe the names when supplied."""

    instructions: Any
    criteria: Mapping[str, Any]


@dataclass(frozen=True)
class Noul:
    """Answer a yes/no question with the probability of ``true``."""

    instructions: Any
    criteria: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Score:
    """Rate on an ordered scale whose first criterion is level zero."""

    instructions: Any
    criteria: Sequence[Any]


Question: TypeAlias = Choice | Noul | Score
