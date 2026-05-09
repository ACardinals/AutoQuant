from __future__ import annotations

import random
from collections.abc import Callable

from market_monitor.models import Candle
from market_monitor.rl.environment import BUY_OR_INCREASE, HOLD, SELL_OR_REDUCE, TradingEnvironment, TradingEnvironmentConfig

Policy = Callable[[dict], int]


def hold_policy(observation: dict) -> int:
    return HOLD


def buy_and_hold_policy() -> Policy:
    bought = False

    def policy(observation: dict) -> int:
        nonlocal bought
        if not bought and observation.get("cash", 0) > 0:
            bought = True
            return BUY_OR_INCREASE
        return HOLD

    return policy


def random_policy(seed: int = 0) -> Policy:
    generator = random.Random(seed)

    def policy(observation: dict) -> int:
        return generator.choice([HOLD, BUY_OR_INCREASE, SELL_OR_REDUCE])

    return policy


def create_policy(name: str, seed: int = 0) -> Policy:
    if name == "hold":
        return hold_policy
    if name == "buy_and_hold":
        return buy_and_hold_policy()
    if name == "random":
        return random_policy(seed)
    raise ValueError(f"Unknown RL baseline policy: {name}")


def available_policies() -> list[str]:
    return ["buy_and_hold", "hold", "random"]


def evaluate_policy(candles: list[Candle], policy: Policy, config: TradingEnvironmentConfig | None = None) -> dict:
    environment = TradingEnvironment(candles, config)
    observation = environment.reset()
    total_reward = 0.0
    steps = 0
    action_counts = {"hold": 0, "buy_or_increase": 0, "sell_or_reduce": 0}

    if environment.current_index >= len(environment.candles) - 1:
        return {
            "total_reward": total_reward,
            "final_equity": round(observation["equity"], 2),
            "steps": steps,
            "action_counts": action_counts,
        }

    while True:
        action = policy(observation)
        result = environment.step(action)
        action = result.info.get("action")
        if action in action_counts:
            action_counts[action] += 1
        total_reward += result.reward
        steps += 1
        observation = result.observation
        if result.done:
            break

    return {
        "total_reward": round(total_reward, 4),
        "final_equity": round(observation["equity"], 2),
        "steps": steps,
        "action_counts": action_counts,
    }
