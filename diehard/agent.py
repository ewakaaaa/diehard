from typing import Dict
from numpy import mean


class Agent:
    def __init__(self, environment) -> None:
        self.env = environment

    def learn(self, epochs: int) -> None:
        pass

    def policy(self, state: str) -> None:
        pass

    def play_single_episode(self, learn: bool) -> Dict:
        pass

    def evaluate(self, epochs: int, learn: bool):
        rewards = []
        for _ in range(epochs):
            self.env.reset()
            rewards.append(self.play_single_episode(learn)["reward"])
        return {"mean": mean(rewards), "max": max(rewards), "min": min(rewards)}
