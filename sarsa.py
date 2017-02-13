import numpy as np

def sarsa(mdp, max_episode, alpha = 0.1, gamma = 0.9, epsilon = 0.1):
    # Initialization
    Q = [[0 for i in range(mdp.A)] for j in range(mdp.S)]
    old_Q = Q
    n_episode = 0
    rewards_per_episode = []
    Q_variances = []
    max_reward = 0
    total_reward = 0
    while n_episode < max_episode:
        try:
            s = mdp.initial_state # Initialize s, starting state
        except AttributeError:
            s = 0
        reward_for_episode = 0
        # With prob epsilon, pick a random action
        if np.random.random_sample() <= epsilon:
            a = np.random.random_integers(0, mdp.A-1)
        else:
            a = np.argmax(Q[s][:])
        while not mdp.is_terminal(s):
            # Observe S and R
            s_new = np.random.choice(range(mdp.S), p = mdp.T[s, a, :])
            r = mdp.R[s_new]
            T_new = np.zeros((mdp.S, mdp.S))

            # Pick new action A' from S'
            if np.random.random_sample() <= epsilon:
                a_new = np.random.random_integers(0, mdp.A-1)
            else:
                a_new = np.argmax(Q[s_new][:])
            Q[s][a] = Q[s][a] + alpha*(r + gamma*Q[s_new][a_new] - Q[s][a])
            s = s_new
            a = a_new
            total_reward += r
            reward_for_episode += r

        if reward_for_episode > max_reward:
            max_reward = reward_for_episode

        rewards_per_episode.append(reward_for_episode)
        Q_variances.append(np.var(Q))

        n_episode += 1
    return Q, total_reward/max_episode, max_reward, rewards_per_episode, Q_variances
