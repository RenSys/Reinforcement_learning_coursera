# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Assignment 2: Optimal Policies with Dynamic Programming
# 
# Welcome to Assignment 2. This notebook will help you understand:
# - Policy Evaluation and Policy Improvement.
# - Value and Policy Iteration.
# - Bellman Equations.
# %% [markdown]
# ## Gridworld City
# 
# Gridworld City, a thriving metropolis with a booming technology industry, has recently experienced an influx of grid-loving software engineers. Unfortunately, the city's street parking system, which charges a fixed rate, is struggling to keep up with the increased demand. To address this, the city council has decided to modify the pricing scheme to better promote social welfare. In general, the city considers social welfare higher when more parking is being used, the exception being that the city prefers that at least one spot is left unoccupied (so that it is available in case someone really needs it). The city council has created a Markov decision process (MDP) to model the demand for parking with a reward function that reflects its preferences. Now the city has hired you &mdash; an expert in dynamic programming &mdash; to help determine an optimal policy.
# %% [markdown]
# ## Preliminaries
# You'll need two imports to complete this assigment:
# - numpy: The fundamental package for scientific computing with Python.
# - tools: A module containing an environment and a plotting function.
# 
# There are also some other lines in the cell below that are used for grading and plotting &mdash; you needn't worry about them.
# 
# In this notebook, all cells are locked except those that you are explicitly asked to modify. It is up to you to decide how to implement your solution in these cells, **but please do not import other libraries** &mdash; doing so will break the autograder.

# %%
# get_ipython().run_cell_magic('capture', '', '%matplotlib inline\nimport numpy as np\nimport pickle\n# import tools')

# %% [markdown]
# In the city council's parking MDP, states are nonnegative integers indicating how many parking spaces are occupied, actions are nonnegative integers designating the price of street parking, the reward is a real value describing the city's preference for the situation, and time is discretized by hour. As might be expected, charging a high price is likely to decrease occupancy over the hour, while charging a low price is likely to increase it.
# 
# For now, let's consider an environment with three parking spaces and three price points. Note that an environment with three parking spaces actually has four states &mdash; zero, one, two, or three spaces could be occupied.

# %%
# %load tools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython import display
import time
import json
from copy import deepcopy

plt.rc('font', size=30)  # controls default text sizes
plt.rc('axes', titlesize=25)  # fontsize of the axes title
plt.rc('axes', labelsize=25)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=17)  # fontsize of the tick labels
plt.rc('ytick', labelsize=17)  # fontsize of the tick labels
plt.rc('legend', fontsize=20)  # legend fontsize
plt.rc('figure', titlesize=30)
plt.tight_layout()


def plot(V, pi):
    # plot value
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
    ax1.axis('on')
    ax1.cla()
    states = np.arange(V.shape[0])
    ax1.bar(states, V, edgecolor='none')
    ax1.set_xlabel('State')
    ax1.set_ylabel('Value', rotation='horizontal', ha='right')
    ax1.set_title('Value Function')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax1.yaxis.grid()
    ax1.set_ylim(bottom=V.min())
    # plot policy
    ax2.axis('on')
    ax2.cla()
    im = ax2.imshow(pi.T, cmap='Greys', vmin=0, vmax=1, aspect='auto')
    ax2.invert_yaxis()
    ax2.set_xlabel('State')
    ax2.set_ylabel('Action', rotation='horizontal', ha='right')
    ax2.set_title('Policy')
    start, end = ax2.get_xlim()
    ax2.xaxis.set_ticks(np.arange(start, end), minor=True)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    start, end = ax2.get_ylim()
    ax2.yaxis.set_ticks(np.arange(start, end), minor=True)
    ax2.grid(which='minor')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.20)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label('Probability', rotation=0, ha='left')
    fig.subplots_adjust(wspace=0.5)
    display.clear_output(wait=True)
    display.display(fig)
    time.sleep(0.001)
    plt.close()


class ParkingWorld:
    def __init__(self,
                 num_spaces=10,
                 num_prices=4,
                 price_factor=0.1,
                 occupants_factor=1.0,
                 null_factor=1 / 3):
        self.__num_spaces = num_spaces
        self.__num_prices = num_prices
        self.__occupants_factor = occupants_factor
        self.__price_factor = price_factor
        self.__null_factor = null_factor
        self.__S = [num_occupied for num_occupied in range(num_spaces + 1)]
        self.__A = list(range(num_prices))

    def state_transition_reward_and_probabilities(self, s, a):
        return np.array([[r, self.p(s_, r, s, a)] for s_, r in self.support(s, a)])

    def support(self, s, a):
        return [(s_, self.reward(s, s_)) for s_ in self.__S]

    def p(self, s_, r, s, a):
        if r != self.reward(s, s_):
            return 0
        else:
            center = (1 - self.__price_factor
                      ) * s + self.__price_factor * self.__num_spaces * (
                             1 - a / self.__num_prices)
            emphasis = np.exp(
                -abs(np.arange(2 * self.__num_spaces) - center) / 5)
            if s_ == self.__num_spaces:
                return sum(emphasis[s_:]) / sum(emphasis)
            return emphasis[s_] / sum(emphasis)

    def reward(self, s, s_):
        return self.state_reward(s) + self.state_reward(s_)

    def state_reward(self, s):
        if s == self.__num_spaces:
            return self.__null_factor * s * self.__occupants_factor
        else:
            return s * self.__occupants_factor

    def random_state(self):
        return np.random.randint(self.__num_prices)

    def step(self, s, a):
        probabilities = [
            self.p(s_, self.reward(s, s_), s, a) for s_ in self.__S
        ]
        return np.random.choice(self.__S, p=probabilities)

    @property
    def A(self):
        return list(self.__A)

    @property
    def num_spaces(self):
        return self.__num_spaces

    @property
    def num_prices(self):
        return self.num_prices

    @property
    def S(self):
        return list(self.__S)


class Transitions(list):
    def __init__(self, transitions):
        self.__transitions = transitions
        super().__init__(transitions)

    def __repr__(self):
        repr = '{:<14} {:<10} {:<10}'.format('Next State', 'Reward',
                                             'Probability')
        repr += '\n'
        for i, (s, r, p) in enumerate(self.__transitions):
            repr += '{:<14} {:<10} {:<10}'.format(s, round(r, 2), round(p, 2))
            if i != len(self.__transitions) - 1:
                repr += '\n'
        return repr


# %%
num_spaces = 3
num_prices = 3
env = ParkingWorld(num_spaces, num_prices)
V = np.zeros(num_spaces + 1)
pi = np.ones((num_spaces + 1, num_prices)) / num_prices

# %% [markdown]
# The value function is a one-dimensional array where the $i$-th entry gives the value of $i$ spaces being occupied.

# %%
V

# %%
state = 0
V[state]

# %%
state = 0
value = 10
V[state] = value
V

# %%
for s, v in enumerate(V):
    print(f'State {s} has value {v}')

# %% [markdown]
# The policy is a two-dimensional array where the $(i, j)$-th entry gives the probability of taking action $j$ in state $i$.

# %%
pi

# %%
state = 0
pi[state]

# %%
state = 0
action = 1
pi[state, action]

# %%
pi[state] = np.array([0.75, 0.21, 0.04])
pi

# %%
for s, pi_s in enumerate(pi):
    print(f''.join(f'pi(A={a}|S={s}) = {p.round(2)}' + 4 * ' ' for a, p in enumerate(pi_s)))

# %%
plot(V, pi)

# %% [markdown]
# We can visualize a value function and policy with the `plot` function in the `tools` module. On the left, the value function is displayed as a barplot. State zero has an expected return of ten, while the other states have an expected return of zero. On the right, the policy is displayed on a two-dimensional grid. Each vertical strip gives the policy at the labeled state. In state zero, action zero is the darkest because the agent's policy makes this choice with the highest probability. In the other states the agent has the equiprobable policy, so the vertical strips are colored uniformly.
# %% [markdown]
# You can access the state space and the action set as attributes of the environment.

# %%
env.S

# %%
env.A

# %% [markdown]
# You will need to use the environment's `transitions` method to complete this assignment. The method takes a state and an action and returns a 2-dimensional array, where the entry at $(i, 0)$ is the reward for transitioning to state $i$ from the current state and the entry at $(i, 1)$ is the conditional probability of transitioning to state $i$ given the current state and action.

# %%
state = 3
action = 1
transitions = env.state_transition_reward_and_probabilities(state, action)
transitions

# %%
for s_, (r, p) in enumerate(transitions):
    print(f'p(S\'={s_}, R={r} | S={state}, A={action}) = {p.round(2)}')


# %% [markdown]
# ## Section 1: Policy Evaluation
# 
# You're now ready to begin the assignment! First, the city council would like you to evaluate the quality of the existing pricing scheme. Policy evaluation works by iteratively applying the Bellman equation for $v_{\pi}$ to a working value function, as an update rule, as shown below.
# 
# $$\large v(s) \leftarrow \sum_a \pi(a | s) \sum_{s', r} p(s', r | s, a)[r + \gamma v(s')]$$
# This update can either occur "in-place" (i.e. the update rule is sequentially applied to each state) or with "two-arrays" (i.e. the update rule is simultaneously applied to each state). Both versions converge to $v_{\pi}$ but the in-place version usually converges faster. **In this assignment, we will be implementing all update rules in-place**, as is done in the pseudocode of chapter 4 of the textbook. 
# 
# We have written an outline of the policy evaluation algorithm described in chapter 4.1 of the textbook. It is left to you to fill in the `bellman_update` function to complete the algorithm.

# %%
def evaluate_policy(env, V, pi, gamma, theta):
    while True:
        delta = 0
        for s in env.S:
            v = V[s]
            bellman_update(env, V, pi, s, gamma)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V


# %%
# [Graded]
def bellman_update(environment, v_state_values, policy_pi, state_idx, gamma_discount):
    """Mutate ``V`` according to the Bellman update equation."""
    ### START CODE HERE ###
    # initalise expected state return i.e. total current and future discounted rewards
    total_state_return = 0
    # loop over all actions
    all_state_actions = environment.A
    for action_idx in all_state_actions:
        probability_of_action = policy_pi[state_idx, action_idx]
        state_transitions = environment.state_transition_reward_and_probabilities(state_idx, action_idx)
        for next_state_idx, (state_reward, state_probability) in enumerate(
                state_transitions):
            action_state_probability = probability_of_action * state_probability
            expect_state_reward = action_state_probability * state_reward

            expected_next_state_future_reward = action_state_probability * v_state_values[next_state_idx]
            discount_future_reward = gamma * expected_next_state_future_reward
            total_state_action_expected_reward = expect_state_reward + discount_future_reward
            total_state_return += total_state_action_expected_reward
    v_state_values[state_idx] = total_state_return
    return v_state_values
    ### END CODE HERE ###


# %% [markdown]
# The cell below uses the policy evaluation algorithm to evaluate the city's policy, which charges a constant price of one.

# %%
# get_ipython().run_line_magic('reset_selective', '-f "^num_spaces$|^num_prices$|^env$|^V$|^pi$|^gamma$|^theta$"')
num_spaces = 10
num_prices = 4
env = ParkingWorld(num_spaces, num_prices)
V = np.zeros(num_spaces + 1)
city_policy = np.zeros((num_spaces + 1, num_prices))
city_policy[:, 1] = 1
gamma = 0.9
theta = 0.1
V = evaluate_policy(env, V, city_policy, gamma, theta)

# %% [markdown]
# You can use the ``plot`` function to visualize the final value function and policy.

# %%
plot(V, city_policy)

# %% [markdown]
# You can check the output (rounded to one decimal place) against the answer below:<br>
# State $\quad\quad$    Value<br>
# 0 $\quad\quad\quad\;$        80.0<br>
# 1 $\quad\quad\quad\;$        81.7<br>
# 2 $\quad\quad\quad\;$        83.4<br>
# 3 $\quad\quad\quad\;$        85.1<br>
# 4 $\quad\quad\quad\;$        86.9<br>
# 5 $\quad\quad\quad\;$        88.6<br>
# 6 $\quad\quad\quad\;$        90.1<br>
# 7 $\quad\quad\quad\;$        91.6<br>
# 8 $\quad\quad\quad\;$        92.8<br>
# 9 $\quad\quad\quad\;$        93.8<br>
# 10 $\quad\quad\;\;\,\,$       87.8<br>
# 
# Observe that the value function qualitatively resembles the city council's preferences &mdash; it monotonically increases as more parking is used, until there is no parking left, in which case the value is lower. Because of the relatively simple reward function (more reward is accrued when many but not all parking spots are taken and less reward is accrued when few or all parking spots are taken) and the highly stochastic dynamics function (each state has positive probability of being reached each time step) the value functions of most policies will qualitatively resemble this graph. However, depending on the intelligence of the policy, the scale of the graph will differ. In other words, better policies will increase the expected return at every state rather than changing the relative desirability of the states. Intuitively, the value of a less desirable state can be increased by making it less likely to remain in a less desirable state. Similarly, the value of a more desirable state can be increased by making it more likely to remain in a more desirable state. That is to say, good policies are policies that spend more time in desirable states and less time in undesirable states. As we will see in this assignment, such a steady state distribution is achieved by setting the price to be low in low occupancy states (so that the occupancy will increase) and setting the price high when occupancy is high (so that full occupancy will be avoided).
# %% [markdown]
# The cell below will check that your code passes the test case above. (Your code passed if the cell runs without error.) Your solution will also be checked against hidden test cases for your final grade. (So don't hard code parameters into your solution.)

# %%
## Test Code for bellman_update() ## 
with open('section1', 'rb') as handle:
    V_correct = pickle.load(handle)
np.testing.assert_array_almost_equal(V, V_correct)


# %% [markdown]
# ## Section 2: Policy Iteration
# Now the city council would like you to compute a more efficient policy using policy iteration. Policy iteration works by alternating between evaluating the existing policy and making the policy greedy with respect to the existing value function. We have written an outline of the policy iteration algorithm described in chapter 4.3 of the textbook. We will make use of the policy evaluation algorithm you completed in section 1. It is left to you to fill in the `q_greedify_policy` function, such that it modifies the policy at $s$ to be greedy with respect to the q-values at $s$, to complete the policy improvement algorithm.

# %%
def improve_policy(env, V, pi, gamma):
    policy_stable = True
    for s in env.S:
        old = pi[s].copy()
        q_greedify_policy(env, V, pi, s, gamma)
        if not np.array_equal(pi[s], old):
            policy_stable = False
    return pi, policy_stable


def policy_iteration(env, gamma, theta):
    V = np.zeros(len(env.S))
    pi = np.ones((len(env.S), len(env.A))) / len(env.A)
    policy_stable = False
    while not policy_stable:
        V = evaluate_policy(env, V, pi, gamma, theta)
        pi, policy_stable = improve_policy(env, V, pi, gamma)
    return V, pi


# %%
# [Graded]
def q_greedify_policy(env, V, pi, s, gamma):
    """Mutate ``pi`` to be greedy with respect to the q-values induced by ``V``."""
    ### START CODE HERE ###
    arr = []
    for a in env.A:
        val = 0
        for s_, (r, p) in enumerate(env.state_transition_reward_and_probabilities(s, a)):
            val += p * (r + gamma * V[s_])
        arr.append(val)
    #     print(f'greedy action {np.argmax(arr)}')

    for a in env.A:
        if a == np.argmax(arr):
            pi[s, a] = 1.0
        else:
            pi[s, a] = 0


### END CODE HERE ###

# %% [markdown]
# When you are ready to test the policy iteration algorithm, run the cell below.

# %%
get_ipython().run_line_magic('reset_selective', '-f "^num_spaces$|^num_prices$|^env$|^V$|^pi$|^gamma$|^theta$"')
env = ParkingWorld(num_spaces=10, num_prices=4)
gamma = 0.9
theta = 0.1
V, pi = policy_iteration(env, gamma, theta)

# %% [markdown]
# You can use the ``plot`` function to visualize the final value function and policy.

# %%
plot(V, pi)

# %% [markdown]
# You can check the value function (rounded to one decimal place) and policy against the answer below:<br>
# State $\quad\quad$    Value $\quad\quad$ Action<br>
# 0 $\quad\quad\quad\;$        81.6 $\quad\quad\;$ 0<br>
# 1 $\quad\quad\quad\;$        83.3 $\quad\quad\;$ 0<br>
# 2 $\quad\quad\quad\;$        85.0 $\quad\quad\;$ 0<br>
# 3 $\quad\quad\quad\;$        86.8 $\quad\quad\;$ 0<br>
# 4 $\quad\quad\quad\;$        88.5 $\quad\quad\;$ 0<br>
# 5 $\quad\quad\quad\;$        90.2 $\quad\quad\;$ 0<br>
# 6 $\quad\quad\quad\;$        91.7 $\quad\quad\;$ 0<br>
# 7 $\quad\quad\quad\;$        93.1 $\quad\quad\;$ 0<br>
# 8 $\quad\quad\quad\;$        94.3 $\quad\quad\;$ 0<br>
# 9 $\quad\quad\quad\;$        95.3 $\quad\quad\;$ 3<br>
# 10 $\quad\quad\;\;\,\,$      89.5 $\quad\quad\;$ 3<br>
# %% [markdown]
# The cell below will check that your code passes the test case above. (Your code passed if the cell runs without error.) Your solution will also be checked against hidden test cases for your final grade. (So don't hard code parameters into your solution.)

# %%
## Test Code for q_greedify_policy() ##
with open('section2', 'rb') as handle:
    V_correct, pi_correct = pickle.load(handle)
np.testing.assert_array_almost_equal(V, V_correct)
np.testing.assert_array_almost_equal(pi, pi_correct)


# %% [markdown]
# ## Section 3: Value Iteration
# The city has also heard about value iteration and would like you to implement it. Value iteration works by iteratively applying the Bellman optimality equation for $v_{\ast}$ to a working value function, as an update rule, as shown below.
# 
# $$\large v(s) \leftarrow \max_a \sum_{s', r} p(s', r | s, a)[r + \gamma v(s')]$$
# We have written an outline of the value iteration algorithm described in chapter 4.4 of the textbook. It is left to you to fill in the `bellman_optimality_update` function to complete the value iteration algorithm.

# %%
def value_iteration(env, gamma, theta):
    V = np.zeros(len(env.S))
    while True:
        delta = 0
        for s in env.S:
            v = V[s]
            bellman_optimality_update(env, V, s, gamma)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    pi = np.ones((len(env.S), len(env.A))) / len(env.A)
    for s in env.S:
        q_greedify_policy(env, V, pi, s, gamma)
    return V, pi


# %%
# [Graded]
def bellman_optimality_update(env, V, s, gamma):
    """Mutate ``V`` according to the Bellman optimality update equation."""
    ### START CODE HERE ###
    arr = []
    # calc the value of each action state
    for a in env.A:
        val = 0
        for s_, (r, p) in enumerate(env.state_transition_reward_and_probabilities(s, a)):
            val += p * (r + gamma * V[s_])
        arr.append(val)
    # update state value = highest state value
    V[s] = max(arr)
    ### END CODE HERE ###


# %% [markdown]
# When you are ready to test the value iteration algorithm, run the cell below.

# %%
get_ipython().run_line_magic('reset_selective', '-f "^num_spaces$|^num_prices$|^env$|^V$|^pi$|^gamma$|^theta$"')
env = ParkingWorld(num_spaces=10, num_prices=4)
gamma = 0.9
theta = 0.1
V, pi = value_iteration(env, gamma, theta)

# %% [markdown]
# You can use the ``plot`` function to visualize the final value function and policy.

# %%
plot(V, pi)

# %% [markdown]
# You can check your value function (rounded to one decimal place) and policy against the answer below:<br>
# State $\quad\quad$    Value $\quad\quad$ Action<br>
# 0 $\quad\quad\quad\;$        81.6 $\quad\quad\;$ 0<br>
# 1 $\quad\quad\quad\;$        83.3 $\quad\quad\;$ 0<br>
# 2 $\quad\quad\quad\;$        85.0 $\quad\quad\;$ 0<br>
# 3 $\quad\quad\quad\;$        86.8 $\quad\quad\;$ 0<br>
# 4 $\quad\quad\quad\;$        88.5 $\quad\quad\;$ 0<br>
# 5 $\quad\quad\quad\;$        90.2 $\quad\quad\;$ 0<br>
# 6 $\quad\quad\quad\;$        91.7 $\quad\quad\;$ 0<br>
# 7 $\quad\quad\quad\;$        93.1 $\quad\quad\;$ 0<br>
# 8 $\quad\quad\quad\;$        94.3 $\quad\quad\;$ 0<br>
# 9 $\quad\quad\quad\;$        95.3 $\quad\quad\;$ 3<br>
# 10 $\quad\quad\;\;\,\,$      89.5 $\quad\quad\;$ 3<br>
# %% [markdown]
# The cell below will check that your code passes the test case above. (Your code passed if the cell runs without error.) Your solution will also be checked against hidden test cases for your final grade. (So don't hard code parameters into your solution.)

# %%
## Test Code for bellman_optimality_update() ## 
with open('section3', 'rb') as handle:
    V_correct, pi_correct = pickle.load(handle)
np.testing.assert_array_almost_equal(V, V_correct)
np.testing.assert_array_almost_equal(pi, pi_correct)


# %% [markdown]
# In the value iteration algorithm above, a policy is not explicitly maintained until the value function has converged. Below, we have written an identically behaving value iteration algorithm that maintains an updated policy. Writing value iteration in this form makes its relationship to policy iteration more evident. Policy iteration alternates between doing complete greedifications and complete evaluations. On the other hand, value iteration alternates between doing local greedifications and local evaluations. 

# %%
def value_iteration2(env, gamma, theta):
    V = np.zeros(len(env.S))
    pi = np.ones((len(env.S), len(env.A))) / len(env.A)
    while True:
        delta = 0
        for s in env.S:
            v = V[s]
            q_greedify_policy(env, V, pi, s, gamma)
            bellman_update(env, V, pi, s, gamma)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V, pi


# %% [markdown]
# You can try the second value iteration algorithm by running the cell below.

# %%
get_ipython().run_line_magic('reset_selective', '-f "^num_spaces$|^num_prices$|^env$|^V$|^pi$|^gamma$|^theta$"')
env = ParkingWorld(num_spaces=10, num_prices=4)
gamma = 0.9
theta = 0.1
V, pi = value_iteration2(env, gamma, theta)
plot(V, pi)

# %% [markdown]
# ## Wrapping Up
# Congratulations, you've completed assignment 2! In this assignment, we investigated policy evaluation and policy improvement, policy iteration and value iteration, and Bellman updates. Gridworld City thanks you for your service!
