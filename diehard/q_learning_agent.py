import random
from agent import Agent
from environment import Environment


class QLearningAgnet(Agent):
    def __init__(self, environment, q_table, alpha, gamma, epsilon) -> None:
        super().__init__(environment)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = q_table
        if len(q_table) == 0:
            self.add_state_to_q_table(self.env.observation_space())

    def add_state_to_q_table(self, state: str) -> None:
        self.q_table[state] = [
            float("-inf") if i == state[0] else 0 for i in self.env.actions
        ]

    def policy(self, state, learn: bool):
        if state[0] != "air":
            return "air"
        if (random.uniform(0, 1) > self.epsilon) or not learn:
            if state in self.q_table:
                row = self.q_table[state]
                index = row.index(max(row))
                return self.env.actions[index]
            else:
                self.add_state_to_q_table(state)
        return random.sample(self.env.action_space(), 1)[0]

    def play_single_episode(self, learn: bool):
        records = []
        cumulative_reward = 0
        done = False
        while not done:
            state = self.env.observation_space()
            action = self.policy(state, learn)
            _, reward, done, next_state = self.env.step(action)
            records.append(next_state)
            if not done:
                cumulative_reward += reward
            if learn:
                # update q-table:
                index_of_action = self.env.actions.index(action)
                old_q_value = self.q_table[state][index_of_action]

                if not next_state in self.q_table:
                    self.add_state_to_q_table(next_state)

                next_max_q_value = max(self.q_table[next_state])
                new_q_value = old_q_value + self.alpha * (
                    reward + self.gamma * next_max_q_value - old_q_value
                )
                self.q_table[state][index_of_action] = new_q_value

        return {"reward": cumulative_reward, "records": records}


if __name__ == "__main__":
    start = random.sample(["fire", "water"], 1)[0]
    env = Environment(start, 100, 100)
    agent = QLearningAgnet(env, {}, 0.2, 0.8, 0.2)
    agent.evaluate(10000, True)
    env.reset()
    print(agent.evaluate(1, True))
