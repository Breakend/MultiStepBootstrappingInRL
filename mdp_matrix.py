import numpy as np

class MDP:
    def __init__(self, T, S, R, A, act_list, terminal_states):
        # State space
        # Integer number of states
        self.S = S

        # Transition probabilities
        # Form: np ndarray of shape (start_state, action, end_state)
        self.T = np.array(T)

        # Reward space
        # Form: vector, rewards for each state
        self.R = np.array(R)

        # Action space
        # integer, number of possible actions
        self.A = A

        # Possible actions in the MDP
        self.actions = act_list

        self.terminal_states = terminal_states

    def is_terminal(self, s):
        return s in self.terminal_states

class GridWorld(MDP):
    def __init__(self, grid_size, reward_pos, terminal_states):
        S = grid_size*grid_size

        R = np.zeros((grid_size, grid_size))

        # Each row of reward_pos is a tuple: x, y, reward
        for row in reward_pos:
            R[row[0], row[1]] = row[2]
        R = R.flatten()

        A = 4
        act_list = ['S', 'E', 'N', 'W']

        T = np.zeros((S, A, S))
        for start_state in range(S):
            state_i = start_state/grid_size
            state_j = (start_state)%grid_size

            # Actions indexed as: 0:S, 1:E, 2:N, 3:W
            for act in range(A):
                feas_grid = np.zeros((grid_size, grid_size))
                if(act == 0 ):
                    if(state_i+1 < grid_size):
                        feas_grid[state_i+1, state_j] = 1
                    else:
                        feas_grid[state_i, state_j] = 1

                elif(act == 1):
                    if(state_j+1 < grid_size):
                        feas_grid[state_i, state_j+1] = 1
                    else:
                        feas_grid[state_i, state_j] = 1

                elif(act == 2):
                    if(state_i-1 >= 0):
                        feas_grid[state_i-1, state_j] = 1
                    else:
                        feas_grid[state_i, state_j] = 1

                elif(act == 3):
                    if(state_j-1 >= 0):
                        feas_grid[state_i, state_j-1] = 1
                    else:
                        feas_grid[state_i, state_j] = 1


                # Flatten the feasibility grid and assign to transition matrix
                T[start_state, act, :] = feas_grid.flatten()
        MDP.__init__(self, T, S, R, A, act_list, terminal_states)


class StochasticGridWorld(MDP):
    def __init__(self, grid_size, reward_pos, terminal_states, p_success = 0.7):
        S = grid_size*grid_size

        R = np.zeros((grid_size, grid_size))

        # Each row of reward_pos is a tuple: x, y, reward
        for row in reward_pos:
            R[row[0], row[1]] = row[2]
        R = R.flatten()

        A = 4
        act_list = ['S', 'E', 'N', 'W']

        T = np.zeros((S, A, S))
        for start_state in range(S):
            state_i = start_state/grid_size
            state_j = (start_state)%grid_size

            # Actions indexed as: 0:S, 1:E, 2:N, 3:W
            for act in range(A):
                feas_grid = np.zeros((grid_size, grid_size))
                if(act == 0 ): # Going South
                    if state_i+1 < grid_size:
                        feas_grid[state_i+1, state_j] = p_success
                    else:
                        feas_grid[state_i, state_j] = p_success

                    if state_j+1 < grid_size:
                        feas_grid[state_i, state_j+1] = (1.0-p_success)/3.0
                    else:
                        feas_grid[state_i, state_j ] += (1.0-p_success)/3.0

                    if state_i-1 >= 0:
                        feas_grid[state_i-1, state_j] = (1.0-p_success)/3.0
                    else:
                        feas_grid[state_i, state_j ] += (1.0-p_success)/3.0

                    if state_j-1 >= 0:
                        feas_grid[state_i, state_j-1] = (1.0-p_success)/3.0
                    else:
                        feas_grid[state_i, state_j ] += (1.0-p_success)/3.0

                elif(act == 1): # Going East
                    if state_j+1 < grid_size:
                        feas_grid[state_i, state_j+1] = p_success
                    else:
                        feas_grid[state_i, state_j] = p_success

                    if state_i+1 < grid_size:
                        feas_grid[state_i+1, state_j] = (1.0-p_success)/3.0
                    else:
                        feas_grid[state_i, state_j ] += (1.0-p_success)/3.0

                    if state_i-1 >= 0:
                        feas_grid[state_i-1, state_j] = (1.0-p_success)/3.0
                    else:
                        feas_grid[state_i, state_j ] += (1.0-p_success)/3.0

                    if state_j-1 >= 0:
                        feas_grid[state_i, state_j-1] = (1.0-p_success)/3.0
                    else:
                        feas_grid[state_i, state_j ] += (1.0-p_success)/3.0

                elif(act == 2): # Going North
                    if state_i-1 >= 0:
                        feas_grid[state_i-1, state_j] = p_success
                    else:
                        feas_grid[state_i, state_j] = p_success

                    if state_j+1 < grid_size:
                        feas_grid[state_i, state_j+1] = (1.0-p_success)/3.0
                    else:
                        feas_grid[state_i, state_j ] += (1.0-p_success)/3.0

                    if state_i+1 < grid_size:
                        feas_grid[state_i+1, state_j] = (1.0-p_success)/3.0
                    else:
                        feas_grid[state_i, state_j ] += (1.0-p_success)/3.0

                    if state_j-1 >= 0:
                        feas_grid[state_i, state_j-1] = (1.0-p_success)/3.0
                    else:
                        feas_grid[state_i, state_j ] += (1.0-p_success)/3.0

                elif(act == 3): # Going West
                    if state_j-1 >= 0:
                        feas_grid[state_i, state_j-1] = p_success
                    else:
                        feas_grid[state_i, state_j] = p_success

                    if state_j+1 < grid_size:
                        feas_grid[state_i, state_j+1] = (1.0-p_success)/3.0
                    else:
                        feas_grid[state_i, state_j ] += (1.0-p_success)/3.0

                    if state_i+1 < grid_size:
                        feas_grid[state_i+1, state_j] = (1.0-p_success)/3.0
                    else:
                        feas_grid[state_i, state_j ] += (1.0-p_success)/3.0

                    if state_i-1 >= 0:
                        feas_grid[state_i-1, state_j] = (1.0-p_success)/3.0
                    else:
                        feas_grid[state_i, state_j ] += (1.0-p_success)/3.0

                # Flatten the feasibility grid and assign to transition matrix
                T[start_state, act, :] = feas_grid.flatten()
        MDP.__init__(self, T, S, R, A, act_list, terminal_states)


class WindyGridCliffMazeWorld(MDP):
    def __init__(self, grid_size, reward_pos, terminal_states, traps, initial_state = [0,0], obstacles = [[i, j, 0] for i in range(10) for j in range(10)]
, p_success = 0.7):
        S = grid_size*grid_size

        R = np.zeros((grid_size, grid_size))

        # Each row of reward_pos is a tuple: x, y, reward
        for row in reward_pos:
            R[row[0], row[1]] = row[2]
        R = R.flatten()

        for i in range(len(traps)):
            if traps[i] == 1:
                R[i] = -25

        # Define actions
        A = 4
        act_list = ['S', 'E', 'N', 'W']

        # Set start state value
        self.initial_state = initial_state[0]*grid_size + initial_state[1]

        T = np.zeros((S, A, S))
        for start_state in range(S):
            state_i = start_state/grid_size
            state_j = (start_state)%grid_size

            # Actions indexed as: 0:S, 1:E, 2:N, 3:W
            for act in range(A):
                feas_grid = np.zeros((grid_size, grid_size))
                if traps[(state_i)*grid_size + state_j] == 1:
                    feas_grid[initial_state[0], initial_state[1]] = 1
                else:
                    if(act == 0 ): # Going South
                        if state_i+1 < grid_size and obstacles[(state_i+1)*grid_size+state_j][2] != 1:
                            feas_grid[state_i+1, state_j] = p_success
                        else:
                            feas_grid[state_i, state_j] = p_success

                        if state_j+1 < grid_size and obstacles[(state_i)*grid_size+state_j+1][2] != 1:
                            feas_grid[state_i, state_j+1] = (1.0-p_success)/3.0
                        else:
                            feas_grid[state_i, state_j ] += (1.0-p_success)/3.0
                        if state_i-1 >= 0 and obstacles[(state_i-1)*grid_size+state_j][2] != 1:
                            feas_grid[state_i-1, state_j] = (1.0-p_success)/3.0
                        else:
                            feas_grid[state_i, state_j ] += (1.0-p_success)/3.0
                        if state_j-1 >= 0 and obstacles[(state_i)*grid_size+state_j-1][2] != 1:
                            feas_grid[state_i, state_j-1] = (1.0-p_success)/3.0
                        else:
                            feas_grid[state_i, state_j ] += (1.0-p_success)/3.0

                    elif(act == 1): # Going East
                        if state_j+1 < grid_size and obstacles[state_i*grid_size+state_j+1][2] != 1:
                            feas_grid[state_i, state_j+1] = p_success
                        else:
                            feas_grid[state_i, state_j] = p_success

                        if state_i+1 < grid_size and obstacles[(state_i+1)*grid_size+state_j][2] != 1:
                            feas_grid[state_i+1, state_j] = (1.0-p_success)/3.0
                        else:
                            feas_grid[state_i, state_j ] += (1.0-p_success)/3.0
                        if state_i-1 >= 0 and obstacles[(state_i-1)*grid_size+state_j][2] != 1:
                            feas_grid[state_i-1, state_j] = (1.0-p_success)/3.0
                        else:
                            feas_grid[state_i, state_j ] += (1.0-p_success)/3.0
                        if state_j-1 >= 0 and obstacles[(state_i)*grid_size+state_j-1][2] != 1:
                            feas_grid[state_i, state_j-1] = (1.0-p_success)/3.0
                        else:
                            feas_grid[state_i, state_j ] += (1.0-p_success)/3.0

                    elif(act == 2): # Going North
                        if state_i-1 >= 0 and obstacles[(state_i-1)*grid_size+state_j][2] != 1:
                            feas_grid[state_i-1, state_j] = p_success
                        else:
                            feas_grid[state_i, state_j] = p_success

                        if state_j+1 < grid_size and obstacles[(state_i)*grid_size+state_j+1][2] != 1:
                            feas_grid[state_i, state_j+1] = (1.0-p_success)/3.0
                        else:
                            feas_grid[state_i, state_j ] += (1.0-p_success)/3.0
                        if state_i+1 < grid_size and obstacles[(state_i+1)*grid_size+state_j][2] != 1:
                            feas_grid[state_i+1, state_j] = (1.0-p_success)/3.0
                        else:
                            feas_grid[state_i, state_j ] += (1.0-p_success)/3.0
                        if state_j-1 >= 0 and obstacles[(state_i)*grid_size+state_j-1][2] != 1:
                            feas_grid[state_i, state_j-1] = (1.0-p_success)/3.0
                        else:
                            feas_grid[state_i, state_j ] += (1.0-p_success)/3.0

                    elif(act == 3): # Going West
                        if state_j-1 >= 0 and obstacles[(state_i)*grid_size+state_j-1][2] != 1:
                            feas_grid[state_i, state_j-1] = p_success
                        else:
                            feas_grid[state_i, state_j] = p_success

                        if state_j+1 < grid_size and obstacles[(state_i)*grid_size+state_j+1][2] != 1:
                            feas_grid[state_i, state_j+1] = (1.0-p_success)/3.0
                        else:
                            feas_grid[state_i, state_j ] += (1.0-p_success)/3.0
                        if state_i+1 < grid_size and obstacles[(state_i+1)*grid_size+state_j][2] != 1:
                            feas_grid[state_i+1, state_j] = (1.0-p_success)/3.0
                        else:
                            feas_grid[state_i, state_j ] += (1.0-p_success)/3.0
                        if state_i-1 >= 0 and obstacles[(state_i-1)*grid_size+state_j][2] != 1:
                            feas_grid[state_i-1, state_j] = (1.0-p_success)/3.0
                        else:
                            feas_grid[state_i, state_j ] += (1.0-p_success)/3.0

                # Flatten the feasibility grid and assign to transition matrix
                T[start_state, act, :] = feas_grid.flatten()
        MDP.__init__(self, T, S, R, A, act_list, terminal_states)