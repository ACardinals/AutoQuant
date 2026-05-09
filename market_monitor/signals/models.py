from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategySignal:
    symbol: str
    action: str
    confidence: float
    reasons: list[str]
    stop_loss: float | None = None
    take_profit: float | None = None
    max_position_pct: float = 0.1

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "signal": self.action,
            "confidence": round(self.confidence, 4),
            "reasons": self.reasons,
            "risk": {
                "stop_loss": self.stop_loss,
                "take_profit": self.take_profit,
                "max_position_pct": self.max_position_pct,
            },
        }
