from agent import Agent
from environment import Environment

# based on https://github.com/GitPistachio/Competitive-programming/blob/master/SPOJ/DIEHARD%20-%20DIE%20HARD/DIE%20HARD.py only to know the correct answer


class ProgrammerAgent(Agent):
    def __init__(self, environment) -> None:
        super().__init__(environment)
        self.solutions = {}

    def policy(self, heath, armor):
        if (heath, armor) in self.solutions:
            return self.solutions[(heath, armor)]
        result = 0
        if heath > 17:
            if armor > 8:
                # You can choose: be on air then of fire or be on air and then on water
                fire = self.policy(heath - 17, armor + 7)
                water = self.policy(heath - 2, armor - 8)
                result = max(fire, water) + 2
            elif armor > 0:
                # You cannot survive to be on air then on water but you can survive be on air then of fire
                fire = self.policy(heath - 17, armor + 7) + 2
                result = fire
        elif heath > 2:
            if armor > 8:
                # You cannot survive to be on air then on water but you can survivie to be on air then of water
                water = self.policy(heath - 2, armor - 8) + 2
                result = water
            elif armor > 0:
                # You only can survive once on air
                result = 1
        elif heath > 0 and armor > 0:
            # You only can survive one on air
            result = 1
            self.solutions[(heath, armor)] = result
        return result

    def play_single_episode(self, learn):
        return {"reward": self.policy(self.env.heath, self.env.armor)}


if __name__ == "__main__":
    env = Environment("air", 500, 600)
    agent = ProgrammerAgent(env)
    print(agent.evaluate(1, False))
