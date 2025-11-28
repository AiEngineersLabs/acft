from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class ConversationTurn:
    user: str
    answer: Optional[str]
    decision: str
    stability: float
    final_phi: Optional[np.ndarray] = None


@dataclass
class ConversationMemory:
    """
    Minimal conversation memory used by the CLI and demos.

    Stores the last N turns plus the final field state (phi) so that
    AttractorVisualizer can compute cosine drift between turns.
    """
    max_turns: int = 10
    turns: List[ConversationTurn] = field(default_factory=list)

    def add_turn(
        self,
        user: str,
        answer: Optional[str],
        decision: str,
        stability: float,
        final_phi: Optional[np.ndarray] = None,
    ) -> None:
        self.turns.append(
            ConversationTurn(
                user=user,
                answer=answer,
                decision=decision,
                stability=stability,
                final_phi=final_phi,
            )
        )
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def build_prompt(self, user_input: str, max_history: int = 3) -> str:
        """
        Build a simple conversational prompt:

        User: ...
        Assistant: ...
        ...

        User: <current input>
        Assistant:
        """
        history = self.turns[-max_history:]
        lines: List[str] = []

        for t in history:
            lines.append(f"User: {t.user}")
            if t.answer is not None:
                lines.append(f"Assistant: {t.answer}")

        lines.append(f"User: {user_input}")
        lines.append("Assistant:")

        return "\n".join(lines)