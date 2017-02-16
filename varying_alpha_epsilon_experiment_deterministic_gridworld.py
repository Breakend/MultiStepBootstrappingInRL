import numpy as np
from mdp_matrix import GridWorld, StochasticGridWorld, WindyGridCliffMazeWorld
from double_sarsa import double_sarsa
from n_step_sarsa import n_step_sarsa
from n_step_expected_sarsa import n_step_expected_sarsa
from n_step_tree_backup import n_step_tree_backup
from q_sigma import n_step_q_sigma
from expected_sarsa import expected_sarsa
from double_expected_sarsa import double_expected_sarsa
import matplotlib.pyplot as plt
from sarsa import sarsa

start_state = [0, 0]

test_rewards = [[i, j, -1] for i in range(5) for j in range(5)]
test_rewards[2] = [0, 2, 1]
test_rewards[23] = [4,3, 1]

start_state = [0, 0]

gw = GridWorld(5, test_rewards, terminal_states=[2, 23] )
# print test_rewards

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


average_reward_n_step_sarsa = []
all_rewards_per_episode_n_step_sarsa = []
q_var_n_step_sarsa = []


average_reward_n_step_expected_sarsa = []
all_rewards_per_episode_n_step_expected_sarsa = []
q_var_n_step_expected_sarsa = []


average_reward_n_step_tree_backup= []
all_rewards_per_episode_n_step_tree_backup = []
q_var_n_step_tree_backup = []

average_reward_n_step_q_sigma= []
all_rewards_per_episode_n_step_q_sigma = []
q_var_n_step_q_sigma = []


epsilon = .1

n=10000
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
        print("Done double sarsa")
        Q, average_reward, max_reward, all_rewards, Q_variances = expected_sarsa(gw, n, epsilon=epsilon, alpha=alpha)
        average_reward_expected_sarsa.append(average_reward)
        q_var_expected_sarsa.append(Q_variances)
        print("Done expected sarsa")
        all_rewards_per_episode_expected_sarsa.append(all_rewards)
        Q, average_reward, max_reward, all_rewards, Q_variances = double_expected_sarsa(gw, n, epsilon=epsilon, alpha=alpha)
        average_reward_double_expected_sarsa.append(average_reward)
        q_var_double_expected_sarsa.append(Q_variances)
        print("Done double expected sarsa")
        all_rewards_per_episode_double_expected_sarsa.append(all_rewards)
        Q, average_reward, max_reward, all_rewards, Q_variances = sarsa(gw, n, epsilon=epsilon, alpha=alpha)
        q_var_sarsa.append(Q_variances)
        print("Done  sarsa")

    n_step_sarsa_results = Parallel(n_jobs=-2, verbose=10)(delayed(n_step_sarsa)(gw, max_episode, alpha, gamma, epsilon, n=4) for alpha in alphas)
    # Q, average_reward, max_reward, all_rewards, Q_variances = n_step_sarsa(gw, max_episode, epsilon=epsilon, alpha=alpha, n = n_step)
    for result in n_step_sarsa_results:
        average_reward_n_step_sarsa.append(result[1])
        q_var_n_step_sarsa.append(result[4])
        all_rewards_per_episode_n_step_sarsa.append(result[3])
    print("Done nstep sarsa")

    n_step_expected_sarsa_results = Parallel(n_jobs=-2, verbose=10)(delayed(n_step_expected_sarsa)(gw, max_episode, alpha, gamma, epsilon, n=4) for alpha in alphas)
    # Q, average_reward, max_reward, all_rewards, Q_variances = n_step_sarsa(gw, max_episode, epsilon=epsilon, alpha=alpha, n = n_step)
    for result in n_step_expected_sarsa_results:
        average_reward_n_step_expected_sarsa.append(result[1])
        q_var_n_step_expected_sarsa.append(result[4])
        all_rewards_per_episode_n_step_expected_sarsa.append(result[3])
    print("Done nstep expected sarsa")

    n_step_tree_backup_results = Parallel(n_jobs=-2, verbose=10)(delayed(n_step_tree_backup)(gw, max_episode, alpha, gamma, epsilon, n) for alpha in alphas)
    # Q, average_reward, max_reward, all_rewards, Q_variances = n_step_tree_backup(gw, max_episode, epsilon=epsilon, alpha=alpha, n = n_step)
    for result in n_step_tree_backup_results:
        average_reward_n_step_tree_backup.append(result[1])
        q_var_n_step_tree_backup.append(result[4])
        all_rewards_per_episode_n_step_tree_backup.append(result[3])
    print("Done nstep tree backup")

    n_step_q_sigma_results = Parallel(n_jobs=-2, verbose=10)(delayed(n_step_q_sigma)(gw, max_episode, alpha, gamma, epsilon, n) for alpha in alphas)
    # Q, average_reward, max_reward, all_rewards, Q_variances = n_step_q_sigma(gw, max_episode, epsilon=epsilon, alpha=alpha, n = n_step)
    for result in n_step_q_sigma_results:
        average_reward_n_step_q_sigma.append(result[1])
        q_var_n_step_q_sigma.append(result[4])
        all_rewards_per_episode_n_step_q_sigma.append(result[3])
    print("Done nstep q_sigma")


q_var_sarsa = np.mean(np.mean(np.split(np.array(q_var_sarsa), number_of_runs), axis = 0), axis=1)
q_var_expected_sarsa = np.mean(np.mean(np.split(np.array(q_var_expected_sarsa), number_of_runs), axis = 0), axis=1)
q_var_double_expected_sarsa = np.mean(np.mean(np.split(np.array(q_var_double_expected_sarsa), number_of_runs), axis = 0), axis=1)
q_var_double_sarsa = np.mean(np.mean(np.split(np.array(q_var_double_sarsa), number_of_runs), axis = 0), axis=1)
q_var_n_step_sarsa = np.mean(np.mean(np.split(np.array(q_var_n_step_sarsa), number_of_runs), axis = 0), axis=1)
q_var_n_step_expected_sarsa = np.mean(np.mean(np.split(np.array(q_var_n_step_expected_sarsa), number_of_runs), axis = 0), axis=1)
q_var_n_step_tree_backup= np.mean(np.mean(np.split(np.array(q_var_n_step_tree_backup), number_of_runs), axis = 0), axis=1)
q_var_n_step_q_sigma= np.mean(np.mean(np.split(np.array(q_var_n_step_q_sigma), number_of_runs), axis = 0), axis=1)


plt.plot(alphas, q_var_double_sarsa, label="Double Sarsa")
plt.plot(alphas, q_var_expected_sarsa, label="Expected Sarsa")
plt.plot(alphas, q_var_double_expected_sarsa, label="Double Expected Sarsa")
plt.plot(alphas, q_var_sarsa, label="Sarsa")
plt.plot(alphas, q_var_n_step_sarsa, label="N-Step Sarsa")
plt.plot(alphas, q_var_n_step_expected_sarsa, label="N-Step Expected Sarsa")
plt.plot(alphas, q_var_n_step_tree_backup, label="N-Step Tree Backup")
plt.plot(alphas, q_var_n_step_q_sigma, label="N-Step Q sigma")

plt.ylabel('Average Q Variance')
plt.xlabel('alpha')
ax = plt.gca()
# ax.set_xscale('symlog')
ax.legend(loc='upper center', shadow=True)
plt.savefig('q_variance.png')
plt.close()

# TODO: plot all sarsa, expected_sarsa, double_Sarsa
# import pdb; pdb.set_trace()
average_reward_double_sarsa = np.mean(np.split(np.array(average_reward_double_sarsa), number_of_runs), axis=0)
average_reward_expected_sarsa = np.mean(np.split(np.array(average_reward_expected_sarsa), number_of_runs), axis=0)
average_reward_double_expected_sarsa = np.mean(np.split(np.array(average_reward_double_expected_sarsa), number_of_runs), axis=0)
average_reward_sarsa = np.mean(np.split(np.array(average_reward_sarsa), number_of_runs), axis=0)
average_reward_n_step_sarsa = np.mean(np.split(np.array(average_reward_n_step_sarsa), number_of_runs), axis=0)
average_reward_n_step_expected_sarsa = np.mean(np.split(np.array(average_reward_n_step_expected_sarsa), number_of_runs), axis=0)
average_reward_n_step_tree_backup = np.mean(np.split(np.array(average_reward_n_step_tree_backup), number_of_runs), axis=0)
average_reward_n_step_q_sigma = np.mean(np.split(np.array(average_reward_n_step_q_sigma), number_of_runs), axis=0)

plt.plot(alphas, average_reward_double_sarsa, label="Double Sarsa")
plt.plot(alphas, average_reward_expected_sarsa, label="Expected Sarsa")
plt.plot(alphas, average_reward_double_expected_sarsa, label="Double Expected Sarsa")
plt.plot(alphas, average_reward_sarsa, label="Sarsa")
plt.plot(alphas, average_reward_n_step_sarsa, label="N-Step Sarsa")
plt.plot(alphas, average_reward_n_step_expected_sarsa, label="N-Step Exepected Sarsa")
plt.plot(alphas, average_reward_n_step_tree_backup, label="N-Step Tree Backup")
plt.plot(alphas, average_reward_n_step_q_sigma, label="N-Step Q Sigma")

plt.ylabel('Average reward')
plt.xlabel('alpha')
ax = plt.gca()
# ax.set_xscale('symlog')
ax.legend(loc='lower center', shadow=True)
plt.savefig('average_reward.png')
plt.close()
# import pdb; pdb.set_trace()

print("Max alpha SARSA: %f" % alphas[np.argmax(average_reward_sarsa)])
print("Max alpha Expected SARSA: %f" % alphas[np.argmax(average_reward_expected_sarsa)])
print("Max alpha Double SARSA: %f" % alphas[np.argmax(average_reward_double_sarsa)])
print("Max alpha Double Expected SARSA: %f" % alphas[np.argmax(average_reward_double_expected_sarsa)])
print("Max alpha N-step SARSA: %f" % alphas[np.argmax(average_reward_n_step_sarsa)])
print("Max alpha N-step Expected SARSA: %f" % alphas[np.argmax(average_reward_n_step_expected_sarsa)])
print("Max alpha N-step TreeBackup: %f" % alphas[np.argmax(average_reward_n_step_tree_backup)])
print("Max alpha N-step Q Sigma: %f" % alphas[np.argmax(average_reward_n_step_q_sigma)])


all_rewards_per_episode_double_sarsa = np.mean(np.split(np.array(all_rewards_per_episode_double_sarsa), number_of_runs), axis=0)
all_rewards_per_episode_expected_sarsa = np.mean(np.split(np.array(all_rewards_per_episode_expected_sarsa), number_of_runs), axis=0)
all_rewards_per_episode_double_expected_sarsa = np.mean(np.split(np.array(all_rewards_per_episode_double_expected_sarsa), number_of_runs), axis=0)
all_rewards_per_episode_sarsa = np.mean(np.split(np.array(all_rewards_per_episode_sarsa), number_of_runs), axis=0)
all_rewards_per_episode_n_step_sarsa = np.mean(np.split(np.array(all_rewards_per_episode_n_step_sarsa), number_of_runs), axis=0)
all_rewards_per_episode_n_step_expected_sarsa = np.mean(np.split(np.array(all_rewards_per_episode_n_step_expected_sarsa), number_of_runs), axis=0)
all_rewards_per_episode_n_step_tree_backup = np.mean(np.split(np.array(all_rewards_per_episode_n_step_tree_backup), number_of_runs), axis=0)
all_rewards_per_episode_n_step_q_sigma= np.mean(np.split(np.array(all_rewards_per_episode_n_step_q_sigma), number_of_runs), axis=0)

variance_double_sarsa = np.var(all_rewards_per_episode_double_sarsa, axis=1)
variance_double_expected_sarsa = np.var(all_rewards_per_episode_double_expected_sarsa, axis=1)
variance_expected_sarsa = np.var(all_rewards_per_episode_expected_sarsa, axis=1)
variance_sarsa = np.var(all_rewards_per_episode_sarsa, axis=1)
variance_n_step_sarsa = np.var(all_rewards_per_episode_n_step_sarsa, axis=1)
variance_n_step_expected_sarsa = np.var(all_rewards_per_episode_n_step_expected_sarsa, axis=1)
variance_n_step_tree_backup = np.var(all_rewards_per_episode_n_step_tree_backup, axis=1)
variance_n_step_q_sigma = np.var(all_rewards_per_episode_n_step_q_sigma, axis=1)

# import pdb; pdb.set_trace()/
plt.plot(alphas, variance_double_sarsa, label="Double Sarsa")
plt.plot(alphas, variance_expected_sarsa, label="Expected Sarsa")
plt.plot(alphas, variance_double_expected_sarsa, label="Double Expected Sarsa")
plt.plot(alphas, variance_sarsa, label="Sarsa")
plt.plot(alphas, variance_n_step_sarsa, label="N-Step Sarsa")
plt.plot(alphas, variance_n_step_expected_sarsa, label="N-Step Expected Sarsa")
plt.plot(alphas, variance_n_step_tree_backup, label="N-Step Tree Backup")
plt.plot(alphas, variance_n_step_q_sigma, label="N-Step Q sigma")

plt.ylabel('Variance in Reward')
plt.xlabel('alpha')
ax = plt.gca()
# ax.set_xscale('symlog')
ax.legend(loc='upper center', shadow=True)
plt.savefig('variance_reward.png')
plt.close()
