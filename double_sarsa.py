import numpy as np

def double_sarsa(mdp, max_episode, alpha = 0.1, gamma = 0.9, epsilon=0.1):
    """
    A simple implementation of double sarsa
    Ganger, Michael, Ethan Duryea, and Wei Hu.
    "Double Sarsa and Double Expected Sarsa with Shallow and Deep Learning."
    Journal of Data Analysis and Information Processing 4.04 (2016): 159.
    """
    # Initialize Q_a, Q_b arbitrarily
    Q_a = [[0 for i in range(mdp.A)] for j in range(mdp.S)]
    Q_b = [[0 for i in range(mdp.A)] for j in range(mdp.S)]

    n_episode = 0
    total_reward = 0
    rewards_per_episode = []
    max_reward = 0
    Q_variances = []

    while n_episode < max_episode:
        # Initialize s, starting state
        try:
            s = mdp.initial_state # Initialize s, starting state
        except AttributeError:
            s = 0

        # With prob epsilon, pick a random action
        if np.random.random_sample() <= epsilon:
            a = np.random.random_integers(0, mdp.A-1)
        else:
            # import pdb; pdb.set_trace()
            a = np.argmax(np.mean([Q_a[s][:], Q_b[s][:]], axis=0))

        r = 0
        reward_for_episode = 0

        while not mdp.is_terminal(s):
            # Observe S and R
            s_new = np.random.choice(range(mdp.S), p = mdp.T[s, a, :])
            r = mdp.R[s_new]
            T_new = np.zeros((mdp.S, mdp.S))

            # Pick new action A' from S'
            if np.random.random_sample() <= epsilon:
                a_new = np.random.random_integers(0, mdp.A-1)
            else:
                a_new = np.argmax(np.mean([Q_a[s_new][:], Q_b[s_new][:]], axis=0))

            Q_a[s][a] = Q_a[s][a] + alpha*(r + gamma*Q_b[s_new][a_new] - Q_a[s][a])
            s = s_new
            a = a_new

            # With some probability .5, swap Q_a and Q_b when performing updates.
            if np.random.random_sample() <= .5:
                tmp = Q_a
                Q_a = Q_b
                Q_b = tmp

            total_reward += r
            reward_for_episode += r

        if reward_for_episode > max_reward:
            max_reward = reward_for_episode

        rewards_per_episode.append(reward_for_episode)
        Q_variances.append(np.var(np.mean([Q_a, Q_b], axis=0)))

        n_episode += 1

    return np.mean([Q_a, Q_b], axis=0), total_reward/max_episode, max_reward, rewards_per_episode, Q_variances
