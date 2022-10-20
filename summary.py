from datetime import datetime
from random_agent import RandomAgent
from q_learning_agent import QLearningAgnet
from programmer_agent import ProgrammerAgent
from environment import Environment

test_cases = [
    {"heath": 1, "armor": 5},
    {"heath": 5, "armor": 80},
    {"heath": 80, "armor": 5},
    {"heath": 100, "armor": 200}
]


def measure(agent, learn, epochs):
    start = datetime.now()
    if learn:
        agent.evaluate(10000, True)
    stats = agent.evaluate(epochs, True)
    end = datetime.now()
    print(f"reward of {agent.__class__.__name__}: {stats}, time: {end - start}")
    env.reset()


for test in test_cases:
    print(f"test case: {test}")
    env = Environment("fire", test["heath"], test["armor"])
    measure(RandomAgent(env), False, 25)
    env.reset()
    measure(QLearningAgnet(env, {}, 0.2, 0.8, 0.2), True, 25)
    env.reset()
    measure(ProgrammerAgent(env), False, 1)
    env.reset()
