import random
from typing import Dict
from agent import Agent
from environment import Environment


class RandomAgent(Agent):
    def __init__(self, environment) -> None:
        super().__init__(environment)

    def policy(self, state: str) -> str:
        if state != "air":
            return "air"
        return random.sample(self.env.action_space(), 1)[0]

    def play_single_episode(self, learn) -> Dict:
        records = []
        cumulative_reward = 0
        done = False
        while not done:
            action = self.policy(self.env.state)
            state, reward, done, info = self.env.step(action)
            records.append(info)
            if not done:
                cumulative_reward += reward
        return {"reward": cumulative_reward, "records": records}


if __name__ == "__main__":
    env = Environment("fire", 100, 100)
    agent = RandomAgent(env)
    print(agent.evaluate(10, False))
    print(agent.__class__.__name__)
