from __future__ import annotations
from typing import Dict, Any


class RoleAgent:
    def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
