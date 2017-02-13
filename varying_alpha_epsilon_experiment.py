import numpy as np
from mdp_matrix import GridWorld, StochasticGridWorld, WindyGridCliffMazeWorld
from double_sarsa import double_sarsa
from expected_sarsa import expected_sarsa
from double_expected_sarsa import double_expected_sarsa
import matplotlib.pyplot as plt
from sarsa import sarsa


# TODO: change these graphs to be over alpha like in the paper

test_rewards = [[i, j, -1.0] for i in range(10) for j in range(10)]
test_rewards[59] = [5, 9, 50]

terminal_states = [59]

obstacles = [[i, j, 0] for i in range(10) for j in range(10)]
obstacles[0*10+3] = [0, 3, 1]
obstacles[0*10+8] = [0, 8, 1]
obstacles[1*10+1] = [1, 1, 1]
obstacles[1*10+4] = [1, 4, 1]
obstacles[1*10+5] = [1, 5, 1]
obstacles[1*10+6] = [1, 6, 1]
obstacles[2*10+1] = [2, 1, 1]
obstacles[2*10+8] = [2, 8, 1]
obstacles[3*10+1] = [3, 1, 1]
obstacles[3*10+4] = [3, 4, 1]
obstacles[3*10+5] = [3, 5, 1]
obstacles[3*10+6] = [3, 6, 1]
obstacles[5*10+0] = [5, 0, 1]
obstacles[5*10+5] = [5, 5, 1]
obstacles[6*10+3] = [6, 3, 1]
obstacles[6*10+4] = [6, 4, 1]
obstacles[6*10+5] = [6, 5, 1]
obstacles[6*10+6] = [6, 6, 1]
obstacles[6*10+8] = [6, 8, 1]
obstacles[7*10+8] = [7, 8, 1]
obstacles[9*10+4] = [9, 4, 1]

traps = [0]*100
traps[4] = 1
traps[9] = 1
traps[13] = 1
traps[33] = 1
traps[43] = 1
traps[47] = 1
traps[67] = 1
traps[72] = 1
traps[96] = 1

start_state = [0, 0]

gw = WindyGridCliffMazeWorld(10, test_rewards, terminal_states, traps, start_state, obstacles)
print test_rewards

average_reward_double_sarsa = []
all_rewards_per_episode_double_sarsa = []
q_var_double_sarsa = []

average_reward_expected_sarsa = []
all_rewards_per_episode_expected_sarsa = []
q_var_expected_sarsa = []

average_reward_double_expected_sarsa = []
all_rewards_per_episode_double_expected_sarsa = []
q_var_double_expected_sarsa = []

average_reward_sarsa = []
all_rewards_per_episode_sarsa = []
q_var_sarsa = []

epsilon = .1

n=1000
alphas = [x for x in np.arange(0.0, 1., .05)]
alphas[0] = .01
# import pdb; pdb.set_trace()

number_of_runs = 5

for r in range(number_of_runs):
    for alpha in alphas:
        print(alpha)
        Q, average_reward, max_reward, all_rewards, Q_variances = double_sarsa(gw, n, epsilon=epsilon, alpha=alpha)
        average_reward_double_sarsa.append(average_reward)
        all_rewards_per_episode_double_sarsa.append(all_rewards)
        q_var_double_sarsa.append(Q_variances)
        Q, average_reward, max_reward, all_rewards, Q_variances = expected_sarsa(gw, n, epsilon=epsilon, alpha=alpha)
        average_reward_expected_sarsa.append(average_reward)
        q_var_expected_sarsa.append(Q_variances)
        all_rewards_per_episode_expected_sarsa.append(all_rewards)
        Q, average_reward, max_reward, all_rewards, Q_variances = double_expected_sarsa(gw, n, epsilon=epsilon, alpha=alpha)
        average_reward_double_expected_sarsa.append(average_reward)
        q_var_double_expected_sarsa.append(Q_variances)
        all_rewards_per_episode_double_expected_sarsa.append(all_rewards)
        Q, average_reward, max_reward, all_rewards, Q_variances = sarsa(gw, n, epsilon=epsilon, alpha=alpha)
        q_var_sarsa.append(Q_variances)
        average_reward_sarsa.append(average_reward)
        all_rewards_per_episode_sarsa.append(all_rewards)


# import pdb; pdb.set_trace()

q_var_sarsa = np.mean(np.mean(np.split(np.array(q_var_sarsa), number_of_runs), axis = 0), axis=1)
q_var_expected_sarsa = np.mean(np.mean(np.split(np.array(q_var_expected_sarsa), number_of_runs), axis = 0), axis=1)
q_var_double_expected_sarsa = np.mean(np.mean(np.split(np.array(q_var_double_expected_sarsa), number_of_runs), axis = 0), axis=1)
q_var_double_sarsa = np.mean(np.mean(np.split(np.array(q_var_double_sarsa), number_of_runs), axis = 0), axis=1)
#
# print("SARSA Mean Q Variance: %f" % np.mean(q_var_sarsa))
# print("Expected SARSA Mean Q Variance: %f" % np.mean(q_var_expected_sarsa))
# print("Double SARSA Mean Q Variance: %f" % np.mean(q_var_double_sarsa))
# print("Double Expected SARSA Mean Q Variance: %f" % np.mean(q_var_double_expected_sarsa))

plt.plot(alphas, q_var_double_sarsa, label="Double Sarsa")
plt.plot(alphas, q_var_expected_sarsa, label="Expected Sarsa")
plt.plot(alphas, q_var_double_expected_sarsa, label="Double Expected Sarsa")
plt.plot(alphas, q_var_sarsa, label="Sarsa")

plt.ylabel('Average Q Variance')
plt.xlabel('alpha')
ax = plt.gca()
# ax.set_xscale('symlog')
ax.legend(loc='upper center', shadow=True)
plt.show()

# TODO: plot all sarsa, expected_sarsa, double_Sarsa
# import pdb; pdb.set_trace()
average_reward_double_sarsa = np.mean(np.split(np.array(average_reward_double_sarsa), number_of_runs), axis=0)
average_reward_expected_sarsa = np.mean(np.split(np.array(average_reward_expected_sarsa), number_of_runs), axis=0)
average_reward_double_expected_sarsa = np.mean(np.split(np.array(average_reward_double_expected_sarsa), number_of_runs), axis=0)
average_reward_sarsa = np.mean(np.split(np.array(average_reward_sarsa), number_of_runs), axis=0)
plt.plot(alphas, average_reward_double_sarsa, label="Double Sarsa")
plt.plot(alphas, average_reward_expected_sarsa, label="Expected Sarsa")
plt.plot(alphas, average_reward_double_expected_sarsa, label="Double Expected Sarsa")
plt.plot(alphas, average_reward_sarsa, label="Sarsa")

plt.ylabel('Average reward')
plt.xlabel('alpha')
ax = plt.gca()
# ax.set_xscale('symlog')
ax.legend(loc='lower center', shadow=True)
plt.show()

# import pdb; pdb.set_trace()

print("Max alpha SARSA: %f" % alphas[np.argmax(average_reward_sarsa)])
print("Max alpha Expected SARSA: %f" % alphas[np.argmax(average_reward_expected_sarsa)])
print("Max alpha Double SARSA: %f" % alphas[np.argmax(average_reward_double_sarsa)])
print("Max alpha Double Expected SARSA: %f" % alphas[np.argmax(average_reward_double_expected_sarsa)])


all_rewards_per_episode_double_sarsa = np.mean(np.split(np.array(all_rewards_per_episode_double_sarsa), number_of_runs), axis=0)
all_rewards_per_episode_expected_sarsa = np.mean(np.split(np.array(all_rewards_per_episode_expected_sarsa), number_of_runs), axis=0)
all_rewards_per_episode_double_expected_sarsa = np.mean(np.split(np.array(all_rewards_per_episode_double_expected_sarsa), number_of_runs), axis=0)
all_rewards_per_episode_sarsa = np.mean(np.split(np.array(all_rewards_per_episode_sarsa), number_of_runs), axis=0)

variance_double_sarsa = np.var(all_rewards_per_episode_double_sarsa, axis=1)
variance_double_expected_sarsa = np.var(all_rewards_per_episode_double_expected_sarsa, axis=1)
variance_expected_sarsa = np.var(all_rewards_per_episode_expected_sarsa, axis=1)
variance_sarsa = np.var(all_rewards_per_episode_sarsa, axis=1)
# import pdb; pdb.set_trace()/
plt.plot(alphas, variance_double_sarsa, label="Double Sarsa")
plt.plot(alphas, variance_expected_sarsa, label="Expected Sarsa")
plt.plot(alphas, variance_double_expected_sarsa, label="Double Expected Sarsa")
plt.plot(alphas, variance_sarsa, label="Sarsa")

plt.ylabel('Variance in Reward')
plt.xlabel('alpha')
ax = plt.gca()
# ax.set_xscale('symlog')
ax.legend(loc='upper center', shadow=True)
plt.show()


#
# for x, e in zip(all_rewards_per_episode_double_sarsa, alphas):
#     # import pdb; pdb.set_trace()
#     plt.plot(x, label="e=%s"%e)
#
#     # break
#
# plt.ylabel('Returns per episode')
# plt.xlabel('episode')
#
# ax = plt.gca()
# # ax.set_xscale('symlog')
# ax.legend(loc='lower right', shadow=True)
# plt.show()
#
#
#
# for x, e in zip(all_rewards_per_episode_expected_sarsa, alphas):
#     # import pdb; pdb.set_trace()
#     plt.plot(x, label="e=%s"%e)
#
#     # break
#
# plt.ylabel('Returns per episode')
# plt.xlabel('episode')
#
# ax = plt.gca()
# # ax.set_xscale('symlog')
# ax.legend(loc='lower right', shadow=True)
# plt.show()
#
# for x, e in zip(all_rewards_per_episode_double_expected_sarsa, alphas):
#     # import pdb; pdb.set_trace()
#     plt.plot(x, label="e=%s"%e)
#
#     # break
#
# plt.ylabel('Returns per episode')
# plt.xlabel('episode')
#
# ax = plt.gca()
# # ax.set_xscale('symlog')
# ax.legend(loc='lower right', shadow=True)
# plt.show()
