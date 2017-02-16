import numpy as np
from mdp_matrix import GridWorld
from n_step_sarsa import n_step_sarsa
from n_step_expected_sarsa import n_step_expected_sarsa
from q_sigma import n_step_q_sigma
from n_step_tree_backup import n_step_tree_backup
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

test_rewards = [[i, j, -1.0] for i in range(5) for j in range(5)]
test_rewards[24] = [4, 4, 10]

terminal_states = [24]

start_state = [0, 0]

gw = GridWorld(5, test_rewards, terminal_states)
print test_rewards

average_reward_n_step_sarsa = []
all_rewards_per_episode_n_step_sarsa = []
q_var_n_step_sarsa = []

average_reward_n_step_tree_backup = []
all_rewards_per_episode_n_step_tree_backup = []
q_var_n_step_tree_backup = []

average_reward_n_step_expected_sarsa = []
all_rewards_per_episode_n_step_expected_sarsa = []
q_var_n_step_expected_sarsa = []

average_reward_qsigma = []
all_rewards_per_episode_qsigma = []
q_var_qsigma = []

epsilon = .1

max_episode=10000
# alphas = [x for x in np.arange(0.0, 1., .05)]
# alphas[0] = .01
alpha = 0.1
gamma = 0.9
n_step_values = [x for x in np.arange(0, 55, 5)]
n_step_values[0] = 1

# import pdb; pdb.set_trace()

number_of_runs = 5

for r in range(number_of_runs):

    # for n_step in n_step_values:
    #     print(n_step)
        # Q, average_reward, max_reward, all_rewards, Q_variances = double_sarsa(gw, n, epsilon=epsilon, alpha=alpha)
        # average_reward_double_sarsa.append(average_reward)
        # all_rewards_per_episode_double_sarsa.append(all_rewards)
        # q_var_double_sarsa.append(Q_variances)
        #
        # Q, average_reward, max_reward, all_rewards, Q_variances = expected_sarsa(gw, n, epsilon=epsilon, alpha=alpha)
        # average_reward_expected_sarsa.append(average_reward)
        # q_var_expected_sarsa.append(Q_variances)
        # all_rewards_per_episode_expected_sarsa.append(all_rewards)
        #
        # Q, average_reward, max_reward, all_rewards, Q_variances = double_expected_sarsa(gw, n, epsilon=epsilon, alpha=alpha)
        # average_reward_double_expected_sarsa.append(average_reward)
        # q_var_double_expected_sarsa.append(Q_variances)
        # all_rewards_per_episode_double_expected_sarsa.append(all_rewards)
        #
        # Q, average_reward, max_reward, all_rewards, Q_variances = sarsa(gw, n, epsilon=epsilon, alpha=alpha)
        # q_var_sarsa.append(Q_variances)
        # average_reward_sarsa.append(average_reward)
        # all_rewards_per_episode_sarsa.append(all_rewards)
    n_step_sarsa_results = Parallel(n_jobs=-2, verbose=10)(delayed(n_step_sarsa)(gw, max_episode, alpha, gamma, epsilon, n) for n in n_step_values)
    # Q, average_reward, max_reward, all_rewards, Q_variances = n_step_sarsa(gw, max_episode, epsilon=epsilon, alpha=alpha, n = n_step)
    for result in n_step_sarsa_results:
        average_reward_n_step_sarsa.append(result[1])
        q_var_n_step_sarsa.append(result[4])
        all_rewards_per_episode_n_step_sarsa.append(result[3])

    n_step_expected_sarsa_results = Parallel(n_jobs=-2, verbose=10)(delayed(n_step_expected_sarsa)(gw, max_episode, alpha, gamma, epsilon, n) for n in n_step_values)
    # Q, average_reward, max_reward, all_rewards, Q_variances = n_step_sarsa(gw, max_episode, epsilon=epsilon, alpha=alpha, n = n_step)
    for result in n_step_expected_sarsa_results:
        average_reward_n_step_expected_sarsa.append(result[1])
        q_var_n_step_expected_sarsa.append(result[4])
        all_rewards_per_episode_n_step_expected_sarsa.append(result[3])

    n_step_tree_backup_results = Parallel(n_jobs=-2, verbose=10)(delayed(n_step_tree_backup)(gw, max_episode, alpha, gamma, epsilon, n) for n in n_step_values)
    # Q, average_reward, max_reward, all_rewards, Q_variances = n_step_tree_backup(gw, max_episode, epsilon=epsilon, alpha=alpha, n = n_step)
    for result in n_step_tree_backup_results:
        average_reward_n_step_tree_backup.append(result[1])
        q_var_n_step_tree_backup.append(result[4])
        all_rewards_per_episode_n_step_tree_backup.append(result[3])

    n_step_q_sigma_results = Parallel(n_jobs=-2, verbose=10)(delayed(n_step_q_sigma)(gw, max_episode, alpha, gamma, epsilon, n) for n in n_step_values)
    # Q, average_reward, max_reward, all_rewards, Q_variances = n_step_q_sigma(gw, max_episode, epsilon=epsilon, alpha=alpha, n = n_step)
    for result in n_step_q_sigma_results:
        average_reward_qsigma.append(result[1])
        q_var_qsigma.append(result[4])
        all_rewards_per_episode_qsigma.append(result[3])

# import pdb; pdb.set_trace()

q_var_n_step_sarsa = np.mean(np.mean(np.split(np.array(q_var_n_step_sarsa), number_of_runs), axis = 0), axis=1)
q_var_n_step_expected_sarsa = np.mean(np.mean(np.split(np.array(q_var_n_step_expected_sarsa), number_of_runs), axis = 0), axis=1)
q_var_n_step_tree_backup = np.mean(np.mean(np.split(np.array(q_var_n_step_tree_backup), number_of_runs), axis = 0), axis=1)
q_var_qsigma = np.mean(np.mean(np.split(np.array(q_var_qsigma), number_of_runs), axis = 0), axis=1)

plt.plot(n_step_values, q_var_n_step_sarsa, label="n-step Sarsa")
plt.plot(n_step_values, q_var_n_step_expected_sarsa, label="n-step expected Sarsa")
plt.plot(n_step_values, q_var_n_step_tree_backup, label="n-step Tree Backup")
plt.plot(n_step_values, q_var_qsigma, label="Q-sigma")

plt.ylabel('Average Q Variance')
plt.xlabel('Number of steps n')
ax = plt.gca()
# ax.set_xscale('symlog')
ax.legend(loc='upper center', shadow=True)
plt.savefig('q_variance.png')


# TODO: plot all sarsa, expected_sarsa, double_Sarsa
# import pdb; pdb.set_trace()
average_reward_n_step_sarsa = np.mean(np.split(np.array(average_reward_n_step_sarsa), number_of_runs), axis=0)
average_reward_n_step_expected_sarsa = np.mean(np.split(np.array(average_reward_n_step_expected_sarsa), number_of_runs), axis=0)
average_reward_n_step_tree_backup = np.mean(np.split(np.array(average_reward_n_step_tree_backup), number_of_runs), axis=0)
average_reward_qsigma = np.mean(np.split(np.array(average_reward_qsigma), number_of_runs), axis=0)
plt.plot(n_step_values, average_reward_n_step_sarsa, label="n-step Sarsa")
plt.plot(n_step_values, average_reward_n_step_expected_sarsa, label="n-step expected Sarsa")
plt.plot(n_step_values, average_reward_n_step_tree_backup, label="n-step Tree Backup")
plt.plot(n_step_values, average_reward_qsigma, label="Q-sigma")

plt.ylabel('Average reward')
plt.xlabel('Number of steps n')
ax = plt.gca()
# ax.set_xscale('symlog')
ax.legend(loc='lower center', shadow=True)
plt.savefig('average_reward.png')

# import pdb; pdb.set_trace()

# print("Max alpha SARSA: %f" % alphas[np.argmax(average_reward_sarsa)])
# print("Max alpha Expected SARSA: %f" % alphas[np.argmax(average_reward_n_step_tree_backup)])
# print("Max alpha Double SARSA: %f" % alphas[np.argmax(average_reward_n_step_sarsa)])
# print("Max alpha Double Expected SARSA: %f" % alphas[np.argmax(average_reward_double_expected_sarsa)])


all_rewards_per_episode_n_step_sarsa = np.mean(np.split(np.array(all_rewards_per_episode_n_step_sarsa), number_of_runs), axis=0)
all_rewards_per_episode_n_step_expected_sarsa = np.mean(np.split(np.array(all_rewards_per_episode_n_step_expected_sarsa), number_of_runs), axis=0)
all_rewards_per_episode_n_step_tree_backup = np.mean(np.split(np.array(all_rewards_per_episode_n_step_tree_backup), number_of_runs), axis=0)
all_rewards_per_episode_qsigma = np.mean(np.split(np.array(all_rewards_per_episode_qsigma), number_of_runs), axis=0)

variance_n_step_sarsa = np.var(all_rewards_per_episode_n_step_sarsa, axis=1)
variance_n_step_expected_sarsa = np.var(all_rewards_per_episode_n_step_expected_sarsa, axis=1)
variance_qsigma = np.var(all_rewards_per_episode_qsigma, axis=1)
variance_n_step_tree_backup = np.var(all_rewards_per_episode_n_step_tree_backup, axis=1)
# import pdb; pdb.set_trace()/
plt.plot(n_step_values, variance_n_step_sarsa, label="n-step Sarsa")
plt.plot(n_step_values, variance_n_step_expected_sarsa, label="n-step expected Sarsa")
plt.plot(n_step_values, variance_n_step_tree_backup, label="n-step Tree Backup")
plt.plot(n_step_values, variance_qsigma, label="Q-sigma")

plt.ylabel('Variance in Reward')
plt.xlabel('Number of steps n')
ax = plt.gca()
# ax.set_xscale('symlog')
ax.legend(loc='upper center', shadow=True)
plt.savefig('variance_reward.png')
