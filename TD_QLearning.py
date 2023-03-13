import numpy as np
import matplotlib.pyplot as plt

# define the state-action value function
# the state-action value function is a list of dictionaries, the length of the list is T
# for each element in the list, it is a dictionary, the key is the state, the value is the dictionary of the action-value function
# for the dictionary of action-value function, the key is the action, the value is the value of the action at this state
# for example, state_action_value_function[0][W0][0.6] is the value of the action 0.6 at the initial state W0
def construct_state_action_value_function(T, W0, a, b, r, actions_space):
    state_action_value_function = [{} for _ in range(T)]
    state_action_value_function[0] = {W0: {i: 0 for i in actions_space}}

    for t in range(1, T):
        for wealth, action_collection in state_action_value_function[t - 1].items():
            W = wealth
            for prev_action in actions_space:
                if int(W + W * prev_action * a + W * (1 - prev_action) * r) not in state_action_value_function[t].keys():
                    state_action_value_function[t][int(W + W * prev_action * a + W * (1 - prev_action) * r)] = {i: 0 for i in actions_space}
                if int(W + W * prev_action * b + W * (1 - prev_action) * r) not in state_action_value_function[
                    t].keys():
                    state_action_value_function[t][int(W + W * prev_action * b + W * (1 - prev_action) * r)] = {i: 0 for i in actions_space}

    return state_action_value_function


# define utility function
def utility_func(x, alpha=0.00005):
    return (1 - np.exp(-alpha*x))/alpha


# find the max value from a dictionary
def find_max_qvalue(qvalue_dict):
    max_value = -np.inf
    max_action = None
    for action, value in qvalue_dict.items():
        if value > max_value:
            max_value = value
            max_action = action
    return max_value, max_action


# Used in the track of convergence, calculate the average value for each time t
def average_value(state_action_value_function, t):
    total_value = 0
    valid_num = 0
    for wealth, action_collection in state_action_value_function[t].items():
        for action, value in action_collection.items():
            if value != 0:
                valid_num += 1
                total_value += value
    return total_value/valid_num


# define a function to perform Q-learning on the state-action value function
def Q_learning(state_action_value_function, T, N, alpha, gamma, epsilon, num_actions, actions_space, W0, a, b, p, r, track_convergence=False):

    convergence_list = [[] for _ in range(10)]

    for i in range(N):
        W = W0
        # update epsilon to perform epsilon-greedy
        if i % (N / 10) == 0:
            epsilon = epsilon * 0.7
        for t in range(T):
            if np.random.rand() < epsilon:
                action = actions_space[np.random.randint(0, num_actions)]
            else:
                qvalue, action = find_max_qvalue(state_action_value_function[t][W])

            if np.random.rand() < p:
                new_W = int(W + W * action * a + W * (1 - action) * r)
            else:
                new_W = int(W + W * action * b + W * (1 - action) * r)

            if t == T - 1:
                # reward = utility_func(new_W) - utility_func(W)
                reward = utility_func(new_W)
                qvalue_revision = reward - state_action_value_function[t][W][action]
                state_action_value_function[t][W][action] += alpha * qvalue_revision

                if track_convergence:
                    for i in range(T):
                        convergence_list[i].append(average_value(state_action_value_function, i))  # record the convergence
                break

            # reward = utility_func(new_W) - utility_func(W)
            reward = utility_func(new_W)
            max_qvalue, next_action = find_max_qvalue(state_action_value_function[t + 1][new_W])
            qvalue_revision = reward + gamma * max_qvalue - state_action_value_function[t][W][action]
            state_action_value_function[t][W][action] += alpha * qvalue_revision
            W = new_W

    if track_convergence:
        return state_action_value_function, convergence_list
    else:
        return state_action_value_function


# test the Q-learning algorithm
if __name__ == '__main__':
    # initialize the parameters, T is the time horizon, W0 is the initial wealth
    T = 10
    W0 = 10000

    # the risky asset parameters, a is the return if going up, b is the return if going down, p is the probability of going up
    a = 0.07
    b = -0.05
    p = 0.7

    # the risk-free asset parameters, r is the return of the risk-free asset
    r = 0.02

    # define the action space, the action space is the percentage of the wealth invested in the risky asset
    num_actions = 5
    actions_space = np.linspace(0, 1, num_actions)
    actions_space = np.round(actions_space, 1)

    # define the parameters for Q-learning
    N = 1000000
    alpha = 0.01
    gamma = 0.1
    epsilon = 1

    # construct the state-action value function
    state_action_value_function = construct_state_action_value_function(T, W0, a, b, r, actions_space)

    # perform Q-learning
    state_action_value_function = Q_learning(state_action_value_function, T, N, alpha, gamma, epsilon, num_actions, actions_space, W0, a, b, p, r)

    # Test optimal policy for 10 random traces
    for i in range(10):
        W = W0
        print('new trace')
        for t in range(T):
            qvalue, action = find_max_qvalue(state_action_value_function[t][W])
            print('t = {}, wealth = {}, action = {}'.format(t, W, action))
            if np.random.rand() < p:
                W = int(W + W * action * a + W * (1 - action) * r)
            else:
                W = int(W + W * action * b + W * (1 - action) * r)

    # Test for convergence, the calculation is very slow since we need to calculate the average value for each time t for each iteration
    # state_action_value_function, convergence_list = Q_learning(state_action_value_function, T, N, alpha, gamma, epsilon, num_actions, actions_space, W0, a, b, p, r, track_convergence=True)
    #
    # # plot the convergence
    # plt.figure(figsize=(10, 8))
    # for i in range(10):
    #     plt.plot(convergence_list[i], label='t = {}'.format(i))
    # plt.legend()
    # plt.xlabel('Number of Iterations')
    # plt.ylabel('Average Value')
    # plt.title('Convergence of Q-learning')
    # plt.show()






