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
    NOTE: taken from https://github.com/dennybritz/reinforcement-learning/blob/master/TD/SARSA.ipynb
    TODO: remove in favour of our own implementation
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


def n_step_sarsa(mdp, max_episode, alpha = 0.1, gamma = 0.9, epsilon = 0.1, n = 10):
    # Initialization
    Q = [[0 for i in range(mdp.A)] for j in range(mdp.S)]
    old_Q = Q
    n_episode = 0
    rewards_per_episode = []
    Q_variances = []
    max_reward = 0
    total_reward = 0

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
        stored_rewards = {}
        stored_states = {}

        # With prob epsilon, pick a random action

        stored_actions[0] = behaviour_policy(Q, s, mdp.A)
        stored_states[0] = s
        reward_for_episode = 0

        while tau < (T-1):
            t += 1
            if t < T:
                # take action A_t

                # Observe and store the next reward R_{t+1} and next state S_{t+1}
                st1 = np.random.choice(range(mdp.S), p = mdp.T[stored_states[t%n], stored_actions[t % n], :])
                # print st1

                rt1 = mdp.R[st1]

                stored_rewards[(t+1) % n] = rt1
                stored_states[(t+1) % n] = st1

                # TODO: is this the right place to put this?
                total_reward += rt1
                reward_for_episode += rt1

                # if s_{t+1} terminal
                if mdp.is_terminal(st1):
                    T = t + 1
                else:
                    stored_actions[(t+1) % n] = behaviour_policy(Q, s, mdp.A)

            tau = t - n + 1 # TODO: +1 here?
            if tau >= 0:
                # product from i = tau+1 to min(tau+n-1, T-1) \pi(A_i | S_i) / \mu(A_i|S_i)
                # import pdb; pdb.set_trace()
                rho = np.prod([target_policy_probs(Q, stored_states[k%n], mdp.A)[stored_actions[k%n]]/behaviour_policy_probs(Q, stored_states[k%n], mdp.A)[stored_actions[k%n]] for k in range(tau+1, min(tau+n-1, T-1)+1)])
                # rho = np.prod([(stored_actions[k % n] == behaviour_policy(Q, stored_states[k% n], mdp.A)) / ((stored_actions[k %n] != np.argmax(Q[stored_states[k%n]][:]))*epsilon*1/mdp.A + (stored_actions[k%n] == np.argmax(Q[stored_states[k%n]][:]))*(epsilon*1/mdp.A + (1-epsilon)))  )])

                # print rho
                G = np.sum([gamma**(i-tau-1) * stored_rewards[i%n] for i in range(tau+1, min(tau+n, T)+1)])



                # print stored_actions
                # print stored_states
                if tau + n < T:
                    G = G + gamma**n * Q[stored_states[(tau+n) %n]][stored_actions[(tau+n) %n]]
                s_tau = stored_states[tau %n]
                a_tau = stored_actions[tau%n]

                Q[s_tau][a_tau] += alpha * rho * (G - Q[s_tau][a_tau])

                # print tau
                # print T
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
