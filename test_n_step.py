import numpy as np
import pprint
from operator import itemgetter
from mdp_matrix import GridWorld, WindyGridCliffMazeWorld, StochasticGridWorld
from sarsa import sarsa
from expected_sarsa import expected_sarsa
from n_step_expected_sarsa import n_step_expected_sarsa
from double_sarsa import double_sarsa
from double_expected_sarsa import double_expected_sarsa
from n_step_sarsa import n_step_sarsa
from n_step_tree_backup import n_step_tree_backup
from q_sigma_with_varianced_sigma import n_step_q_sigma

start_state = [0, 0]

test_rewards = [[i, j, -1] for i in range(5) for j in range(5)]
test_rewards[2] = [0, 2, 1]
test_rewards[23] = [4,3, 1]

def transform_to_actions(f):
    actions = ["S", "E", "N", "W"]
    return np.array([actions[x] for x in f])


rewards = np.ones((5,5))
for x,y,r in test_rewards:
    rewards[x][y] = r

gw = GridWorld(5, test_rewards, terminal_states=[2, 23] )

# print gw.T

# Q, ave_reward, max_reward, rewards_per_episode, Q_variances = n_step_sarsa(gw, 20000, alpha=.1, n=10)
Q, ave_reward, max_reward, rewards_per_episode, Q_variances = n_step_q_sigma(gw, 20000, alpha=.7, n=4)
sarsa_Q, ave_reward, max_reward, rewards_per_episode, Q_variances = sarsa(gw, 20000, alpha=.1)


print "REWARDS"
print np.reshape(np.array(rewards), (5,5))

print np.reshape(transform_to_actions(np.argmax(Q, 1)), (5,5))

print "SARSA"
print np.reshape(transform_to_actions(np.argmax(sarsa_Q, 1)), (5,5))
