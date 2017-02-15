import numpy as np
from mdp_matrix import GridWorld, StochasticGridWorld, WindyGridCliffMazeWorld
from double_sarsa import double_sarsa
from n_step_sarsa import n_step_sarsa
from n_step_expected_sarsa import n_step_expected_sarsa
from expected_sarsa import expected_sarsa
from double_expected_sarsa import double_expected_sarsa
import matplotlib.pyplot as plt
from sarsa import sarsa

start_state = [0, 0]

test_rewards = [[i, j, -1] for i in range(10) for j in range(10)]
test_rewards[2] = [0, 2, 1]
test_rewards[23] = [4,3, 1]

start_state = [0, 0]

gw = GridWorld(10, test_rewards, terminal_states=[2, 23] )
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
        Q, average_reward, max_reward, all_rewards, Q_variances = n_step_sarsa(gw, n, epsilon=epsilon, alpha=alpha)
        q_var_n_step_sarsa.append(Q_variances)
        average_reward_n_step_sarsa.append(average_reward)
        all_rewards_per_episode_n_step_sarsa.append(all_rewards)
        Q, average_reward, max_reward, all_rewards, Q_variances = n_step_expected_sarsa(gw, n, epsilon=epsilon, alpha=alpha)
        q_var_n_step_expected_sarsa.append(Q_variances)
        average_reward_n_step_expected_sarsa.append(average_reward)
        all_rewards_per_episode_n_step_expected_sarsa.append(all_rewards)



q_var_sarsa = np.mean(np.mean(np.split(np.array(q_var_sarsa), number_of_runs), axis = 0), axis=1)
q_var_expected_sarsa = np.mean(np.mean(np.split(np.array(q_var_expected_sarsa), number_of_runs), axis = 0), axis=1)
q_var_double_expected_sarsa = np.mean(np.mean(np.split(np.array(q_var_double_expected_sarsa), number_of_runs), axis = 0), axis=1)
q_var_double_sarsa = np.mean(np.mean(np.split(np.array(q_var_double_sarsa), number_of_runs), axis = 0), axis=1)
q_var_n_step_sarsa = np.mean(np.mean(np.split(np.array(q_var_n_step_sarsa), number_of_runs), axis = 0), axis=1)
q_var_n_step_expected_sarsa = np.mean(np.mean(np.split(np.array(q_var_n_step_expected_sarsa), number_of_runs), axis = 0), axis=1)


plt.plot(alphas, q_var_double_sarsa, label="Double Sarsa")
plt.plot(alphas, q_var_expected_sarsa, label="Expected Sarsa")
plt.plot(alphas, q_var_double_expected_sarsa, label="Double Expected Sarsa")
plt.plot(alphas, q_var_sarsa, label="Sarsa")
plt.plot(alphas, q_var_n_step_sarsa, label="N-Step Sarsa")
plt.plot(alphas, q_var_n_step_expected_sarsa, label="N-Step Expected Sarsa")

plt.ylabel('Average Q Variance')
plt.xlabel('alpha')
ax = plt.gca()
# ax.set_xscale('symlog')
ax.legend(loc='upper center', shadow=True)
plt.savefig('q_variance.png')


# TODO: plot all sarsa, expected_sarsa, double_Sarsa
# import pdb; pdb.set_trace()
average_reward_double_sarsa = np.mean(np.split(np.array(average_reward_double_sarsa), number_of_runs), axis=0)
average_reward_expected_sarsa = np.mean(np.split(np.array(average_reward_expected_sarsa), number_of_runs), axis=0)
average_reward_double_expected_sarsa = np.mean(np.split(np.array(average_reward_double_expected_sarsa), number_of_runs), axis=0)
average_reward_sarsa = np.mean(np.split(np.array(average_reward_sarsa), number_of_runs), axis=0)
average_reward_n_step_sarsa = np.mean(np.split(np.array(average_reward_n_step_sarsa), number_of_runs), axis=0)
average_reward_n_step_expected_sarsa = np.mean(np.split(np.array(average_reward_n_step_expected_sarsa), number_of_runs), axis=0)

plt.plot(alphas, average_reward_double_sarsa, label="Double Sarsa")
plt.plot(alphas, average_reward_expected_sarsa, label="Expected Sarsa")
plt.plot(alphas, average_reward_double_expected_sarsa, label="Double Expected Sarsa")
plt.plot(alphas, average_reward_sarsa, label="Sarsa")
plt.plot(alphas, average_reward_n_step_sarsa, label="N-Step Sarsa")
plt.plot(alphas, average_reward_n_step_expected_sarsa, label="N-Step Exepected Sarsa")

plt.ylabel('Average reward')
plt.xlabel('alpha')
ax = plt.gca()
# ax.set_xscale('symlog')
ax.legend(loc='lower center', shadow=True)
plt.savefig('average_reward.png')

# import pdb; pdb.set_trace()

print("Max alpha SARSA: %f" % alphas[np.argmax(average_reward_sarsa)])
print("Max alpha Expected SARSA: %f" % alphas[np.argmax(average_reward_expected_sarsa)])
print("Max alpha Double SARSA: %f" % alphas[np.argmax(average_reward_double_sarsa)])
print("Max alpha Double Expected SARSA: %f" % alphas[np.argmax(average_reward_double_expected_sarsa)])
print("Max alpha N-step SARSA: %f" % alphas[np.argmax(average_reward_n_step_sarsa)])
print("Max alpha N-step Expected SARSA: %f" % alphas[np.argmax(average_reward_n_step_expected_sarsa)])


all_rewards_per_episode_double_sarsa = np.mean(np.split(np.array(all_rewards_per_episode_double_sarsa), number_of_runs), axis=0)
all_rewards_per_episode_expected_sarsa = np.mean(np.split(np.array(all_rewards_per_episode_expected_sarsa), number_of_runs), axis=0)
all_rewards_per_episode_double_expected_sarsa = np.mean(np.split(np.array(all_rewards_per_episode_double_expected_sarsa), number_of_runs), axis=0)
all_rewards_per_episode_sarsa = np.mean(np.split(np.array(all_rewards_per_episode_sarsa), number_of_runs), axis=0)
all_rewards_per_episode_n_step_sarsa = np.mean(np.split(np.array(all_rewards_per_episode_n_step_sarsa), number_of_runs), axis=0)
all_rewards_per_episode_n_step_expected_sarsa = np.mean(np.split(np.array(all_rewards_per_episode_n_step_expected_sarsa), number_of_runs), axis=0)

variance_double_sarsa = np.var(all_rewards_per_episode_double_sarsa, axis=1)
variance_double_expected_sarsa = np.var(all_rewards_per_episode_double_expected_sarsa, axis=1)
variance_expected_sarsa = np.var(all_rewards_per_episode_expected_sarsa, axis=1)
variance_sarsa = np.var(all_rewards_per_episode_sarsa, axis=1)
variance_n_step_sarsa = np.var(all_rewards_per_episode_n_step_sarsa, axis=1)
variance_n_step_expected_sarsa = np.var(all_rewards_per_episode_n_step_expected_sarsa, axis=1)

# import pdb; pdb.set_trace()/
plt.plot(alphas, variance_double_sarsa, label="Double Sarsa")
plt.plot(alphas, variance_expected_sarsa, label="Expected Sarsa")
plt.plot(alphas, variance_double_expected_sarsa, label="Double Expected Sarsa")
plt.plot(alphas, variance_sarsa, label="Sarsa")
plt.plot(alphas, variance_n_step_sarsa, label="N-Step Sarsa")
plt.plot(alphas, variance_n_step_expected_sarsa, label="N-Step Expected Sarsa")

plt.ylabel('Variance in Reward')
plt.xlabel('alpha')
ax = plt.gca()
# ax.set_xscale('symlog')
ax.legend(loc='upper center', shadow=True)
plt.savefig('variance_reward.png')
