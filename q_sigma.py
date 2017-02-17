import numpy as np
import sys

def behaviour_policy(Q, s, nA, epsilon=.3):
    """
    Recall that off-policy learning is learning the value function for
    one policy, \pi, while following another policy, \mu. Often, \pi is
    the greedy policy for the current action-value-function estimate,
    and \mu is a more exploratory policy, perhaps \epsilon-greedy.
    In order to use the data from \pi we must take into account the
    difference between the two policies, using their relative
    probability of taking the actions that were taken.
    NOTE: some parts taken from https://github.com/dennybritz/reinforcement-learning/blob/master/TD/SARSA.ipynb

    """
    A = behaviour_policy_probs(Q, s, nA, epsilon)
    return np.random.choice(range(nA),p= A)

def behaviour_policy_probs(Q, s, nA, epsilon=.3):
    A = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[s][:])
    A[best_action] += (1.0 - epsilon)
    return A

def target_policy(Q, s, nA, epsilon=0):
    A = target_policy_probs(Q, s, nA, epsilon)
    return np.random.choice(range(nA),p= A)

def target_policy_probs(Q, s, nA, epsilon=.1):
    A = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[s][:])
    A[best_action] += (1.0 - epsilon)
    return A

def select_sigma(Q, s, a, nA):
    """
    The idea is that you should be able to choose a sigma based off of a function of
    variables, but here we simply choose a uniform distribution for now.
    Idea is to swap this out for other functions in experiments
    """
    return np.random.randint(2, size=nA)[a]


def n_step_q_sigma(mdp, max_episode, alpha = 0.1, gamma = 0.9, epsilon = 0.1, n = 10):
    # Initialization
    Q = [[0 for i in range(mdp.A)] for j in range(mdp.S)]
    n_episode = 0
    rewards_per_episode = []
    Q_variances = []
    max_reward = 0
    total_reward = 0

    # TODO: make sigma able to be configured

    while n_episode < max_episode:
        # If there's no starting state, just start at state 0
        try:
            s = mdp.initial_state # Initialize s, starting state
        except AttributeError:
            s = 0

        # initializations
        T = sys.maxint
        tau = 0
        t = -1
        stored_actions = {}
        stored_deltas = {}
        stored_states = {}
        stored_Qs = {}
        stored_bp = {}
        stored_rhos = {} #importance sampling ratios for backups
        stored_sigmas = {}

        # With prob epsilon, pick a random action
        stored_actions[0] = behaviour_policy(Q, s, mdp.A)
        stored_states[0] = s
        stored_Qs[0] = Q[s][stored_actions[0]]
        stored_bp[0] = behaviour_policy_probs(Q, s, mdp.A)[stored_actions[0]]
        stored_sigmas[0] = select_sigma(Q, s, stored_actions[0], mdp.A)
        stored_rhos[0] = target_policy_probs(Q, s, mdp.A)[stored_actions[0]] / behaviour_policy_probs(Q, s, mdp.A)[stored_actions[0]]

        for i in range(1, n):
            stored_bp[i] = 0.
        for i in range(1, n):
            stored_sigmas[i] = 0.
        reward_for_episode = 0

        while tau < (T-1):
            t += 1
            if t < T:
                # Take action A_t
                # Observe and store the next reward R_{t+1} and next state S_{t+1}
                st1 = np.random.choice(range(mdp.S), p = mdp.T[stored_states[t%n], stored_actions[t % n], :])
                r = mdp.R[st1]

                stored_states[(t+1) % n] = st1

                # TODO: is this the right place to put this?
                total_reward += r
                reward_for_episode += r

                # if s_{t+1} terminal
                if mdp.is_terminal(st1):
                    T = t + 1
                    stored_deltas[t%n] = r - stored_Qs[t%(n+1)]
                else:
                    at1 = behaviour_policy(Q, s, mdp.A)
                    stored_actions[(t+1) % n] = at1

                    # select and store sigma
                    stored_sigmas[(t+1)%n] = sigma = select_sigma(Q, st1, at1, mdp.A)

                    # Store Q(st1|At1)
                    stored_Qs[(t+1) % (n+1)] = Q[st1][at1]
                    # Store R + ... as sigma_t
                    expectedQ = sum([behaviour_policy_probs(Q, st1, mdp.A)[a]*Q[st1][a] for a in range(mdp.A)])
                    stored_deltas[t%n] = r + gamma * sigma * stored_Qs[(t+1)%(n+1)] + gamma*(1-sigma)* expectedQ - stored_Qs[t%(n+1)]

                    # Select arbitrarily and store and action as A_t+1

                    # print "tau " + str(tau)
                    # print "t :" + str(t)
                    # print "T : " + str(T)
                    # Store behaviour policy as pi_t1
                    stored_bp[(t+1) % n] = behaviour_policy_probs(Q, st1, mdp.A)[at1]
                    stored_rhos[(t+1) % n] = target_policy_probs(Q, st1, mdp.A)[at1] / behaviour_policy_probs(Q, st1, mdp.A)[at1]


            tau = t - n + 1 # TODO: +1 here?
            if tau >= 0:
                E = 1.0
                rho = 1.0
                G = stored_Qs[tau % (n+1)]
#                 print "first:" + str(tau+n-1)
#                 print "second:" + str(T-1)

                for k in range(tau, min(tau+n-1, T-1)+1):
                    # NON ONLINE VERSION
                    # if k >= tau + 1:
                    #     E = np.prod([ gamma* (stored_sigmas[l%n] * stored_bp[(l%n)] + stored_sigmas[l%n]) for l in range(tau+1, k+1)])
                    # if E == 0:
                    #     E = 1
                    # G = G + E * stored_deltas[k%n]
                    # rho *= (1 - stored_sigmas[k%n] + stored_sigmas[k%n]*stored_rhos[k%n])

                    G = G + E * stored_deltas[k%n]
                    factor = (1 - stored_sigmas[(k+1)%n]) * stored_bp[((k+1)%n)] + stored_sigmas[(k+1)%n]
                    # print factor
                    E *= gamma * factor
                    # if k >= tau + 1:
                    #     E = np.prod([ gamma*  for l in range(tau+1, k+1)])
                    if E == 0:
                        E = 1
                    rho *= (1 - stored_sigmas[k%n] + stored_sigmas[k%n]*stored_rhos[k%n])
                    # E = gamma * E * stored_bp[(k+1)%n]

                s_tau = stored_states[tau%n]
                a_tau = stored_actions[tau%n]
                # rho = np.prod([ (1- stored_sigmas[k]) for k in range(tau+1, min(tau+n, T-1)+1)])

                Q[s_tau][a_tau] += alpha * rho * (G - Q[s_tau][a_tau])
                # if pi is being learned, ensure that pi(.|S_tau) is \epsilon-greedy wrt Q

        if reward_for_episode > max_reward:
            max_reward = reward_for_episode

        rewards_per_episode.append(reward_for_episode)
        Q_variances.append(np.var(Q))

        #TODO: should we instead do an on-policy run here to calculate the
        # average reward for the episode?

        n_episode += 1
        # print "Episode: %d" % n_episode
    return Q, total_reward/max_episode, max_reward, rewards_per_episode, Q_variances
