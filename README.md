# diehard
Solving the DIEHARD - DIE HARD problem (https://www.spoj.com/problems/DIEHARD/) by reinforcement learning.

The only intended policy of an agent is to go to "air" from another state. The decision about which state to go into from the "air" in the case of a random agent is random. In the case of the q-agent, the information is taken from the Q-matrix representing the relationship between the next step and reward of doing this step.

Process contains random elements (epsilon value when we allow q-agent to randomly act). Therefore, your results may be different.

Examples of q-agent performance quality checks (summary.py): 
```bash
test case: {'heath': 1, 'armor': 5, 'true': 1}
reward of RandomAgent: {'mean': 1.0, 'max': 1, 'min': 1}, time: 0:00:00.000810
reward of QLearningAgnet: {'mean': 1.0, 'max': 1, 'min': 1}, time: 0:00:00.090506
test case: {'heath': 18, 'armor': 18, 'true': 5}
reward of RandomAgent: {'mean': 3.48, 'max': 5, 'min': 3}, time: 0:00:00.000555
reward of QLearningAgnet: {'mean': 4.84, 'max': 5, 'min': 3}, time: 0:00:00.224724
test case: {'heath': 5, 'armor': 80, 'true': 5}
reward of RandomAgent: {'mean': 2.6, 'max': 5, 'min': 1}, time: 0:00:00.000451
reward of QLearningAgnet: {'mean': 4.84, 'max': 5, 'min': 1}, time: 0:00:00.228023
test case: {'heath': 100, 'armor': 200, 'true': 57}
reward of RandomAgent: {'mean': 22.04, 'max': 39, 'min': 13}, time: 0:00:00.002897
reward of QLearningAgnet: {'mean': 56.44, 'max': 57, 'min': 55}, time: 0:00:02.202587
test case: {'heath': 500, 'armor': 600, 'true': 217}
reward of RandomAgent: {'mean': 105.48, 'max': 123, 'min': 79}, time: 0:00:00.015272
reward of QLearningAgnet: {'mean': 205.4, 'max': 217, 'min': 169}, time: 0:00:08.041461
test case: {'heath': 1000, 'armor': 1000, 'true': 399}
reward of RandomAgent: {'mean': 210.44, 'max': 243, 'min': 189}, time: 0:00:00.026674
reward of QLearningAgnet: {'mean': 356.44, 'max': 399, 'min': 299}, time: 0:00:14.008877
```
Conclusions:
- The larger the initial values, the worse the agent does, therefore the process gets longer and we assign the value of the reward to individual steps worse and worse (reward - taking a step)
- No less, the maximum reward value always equals the real number of steps needed, so the agent has learned how to maximize the survival time! 
- For the correct operation, it was necessary to add an earlier stop of the learning process to be able to save the optimal state of the Q matrix