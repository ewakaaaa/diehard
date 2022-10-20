class Environment:
    def __init__(self, state, heath, armor) -> None:
        self.reward = 1  # reward for taking steps
        self.actions = ["air", "water", "fire"]
        self.state = state
        self.heath = heath
        self.armor = armor
        self.start_state = state
        self.start_heath = heath
        self.start_armor = armor

    def reset(self):
        self.state = self.start_state
        self.heath = self.start_heath
        self.armor = self.start_armor

    def check_if_done(self):
        if self.heath <= 0 or self.armor <= 0:
            return True
        return False

    def update_heath_and_armor(self, action):
        if action == "air":
            self.heath = self.heath + 3
            self.armor = self.armor + 2
        if action == "fire":
            self.heath = self.heath - 20
            self.armor = self.armor + 5
        if action == "water":
            self.heath = self.heath - 5
            self.armor = self.armor - 10

    def observation_space(self):
        return (self.state, self.heath, self.armor)

    def action_space(self):
        return set(self.actions) - {self.state}

    def step(self, action):
        self.state = action
        self.update_heath_and_armor(action)
        return action, self.reward, self.check_if_done(), self.observation_space()
