import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


class RL(object):
    # Implements policy evaluation and optimization
    def __init__(self, env):
        """
        :param env: MDP simulator

        Addtionnal attributes:
             mu: starting state distribution estimation with MC samples
             policy: policy we want to evaluate with estimate_value_MC #(n_states)
             policy_n: records of each episode greedy policy in the Q-learning process #(n_states, n_iterations + 1)
             value_n: records of each episode value function estimate (in policy evaluation: vf of the selected policy
                                                                     // in policy optimization: vf of greedy policy)
                                                                     #(n_states, n_iterations + 1)
            avg_value_n: records of each episode expected value obtained when playing the game
                         (taking into account starting state distribution) #(n_iterations + 1)
            alpha: learning rates functions #(n_states, n_actions)
            Q_n: records of each episode Q-function #(n_states, n_actions, n_iterations + 1)
            reward_n: records of each episode cumulated reward #(n_iterations + 1)
        """

        self.env = env

        self.mu = None

        self.policy = None
        self.policy_n = None

        self.value_n = None
        self.avg_value_n = None
        self.avg_value_infty = None

        self.alpha = None
        self.Q_n = None

        self.reward_n = None
        self.disct_reward_n = None

    def estimate_start_distribution(self, n_start):
        """
        :return: Estimates start state distribution with empirical frequency by resetting n_start times the env
        """

        self.mu = np.zeros(self.env.n_states)
        for i in range(n_start):
            start_pos = self.env.reset()
            self.mu[start_pos] += 1

        self.mu = self.mu / np.sum(self.mu)

    def set_policy(self, policy):
        """
        :return: Sets custom policy
        """
        self.policy = policy

    def estimate_value_MC(self, n, tmax):
        """
        :return: Evaluates policy with first-initial-state MC estimator (with n sample trajectory and cutting each trajectory after tmax steps or reaching terminal states)
        """

        self.value_n = np.zeros((self.env.n_states, n+1))
        # occurrences keep track of many times we visited each state #(n_states)
        occurrences = np.zeros(self.env.n_states)

        # n episodes ran
        for i in range(n):
            reached_terminal = False
            start_pos = self.env.reset()
            state = start_pos
            traj_reward = 0
            weight = 1
            occurrences[start_pos] += 1
            t = 0

            # we stop after tmax step if we didn't reach terminal state yet
            while (not reached_terminal) and t < tmax:
                action = self.policy[state]
                state, reward, reached_terminal = self.env.step(state, action)

                traj_reward += weight*reward
                weight = self.env.gamma*weight
                t += 1

            # recursive relationship to compute new estimate of value function based on previous estimate with the new sample
            # as we visit more and more a state, the marginal contribution is lowered due to occurences factor
            self.value_n[start_pos,i+1] = ((occurrences[start_pos] - 1)*self.value_n[start_pos, i] + traj_reward)/occurrences[start_pos]
            for pos in range(self.env.n_states):
                if pos != start_pos:
                    self.value_n[pos, i+1] = self.value_n[pos, i]

    def plot_avg_value_approximation(self, value_infty, label = '', ax = None):
        """
        :return: plot average value of the game estimation error in terms of number of iterations
        """
        # computing average value of the game
        self.avg_value_n = (self.mu.reshape((self.env.n_states,1)) * self.value_n).sum(axis = 0)

        self.avg_value_infty = (value_infty * self.mu).sum()
        # print(f"Optimal value of the game with policy {self.policy}: {self.avg_value_infty}")
        data_plot = self.avg_value_n - self.avg_value_infty

        yline = 0
        title = r'$J_n - J^\pi$ in terms of number of iterations'
        ylabel = r'$J_n - J^\pi$'
        xlabel = 'Number of iterations n'

        self.plot_standard(data_plot, label, yline, title, ylabel, xlabel, ax)

    def _process_NA_actions(self):
        """
        :return: fill non-available actions of each state in Q-function by -np.inf
        Remark: -np.inf choice is relevant as we try to take the maximum of Q_n function at each step - thus we should never select a non-available action
        """
        for state in range(self.env.n_states):
            available_actions = self.env.state_actions[state]
            NA_actions = [x for x in np.arange(0,len(self.env.action_names)) if x not in available_actions]
            self.Q_n[state, NA_actions, :] = -np.inf

    def learn_q(self, n, tmax, eps, alpha):
        """
        :return: Learn Q-function with Q-learning method and extract greedy policy out of the final Q-function
        """

        self.alpha = alpha
        self.Q_n = np.zeros((self.env.n_states, len(self.env.action_names),n+1))
        occurrences = np.zeros((self.env.n_states, len(self.env.action_names)))
        self.disct_reward_n = np.zeros(n+1)
        self.reward_n = np.zeros(n+1)

        # Processing non-available actions
        self._process_NA_actions()

        # Run n episodes
        for i in range(n):
            reached_terminal = False
            start_pos = self.env.reset()
            state = start_pos
            reward_i = 0
            disct_reward_i = 0
            weight = 1

            t = 0

            self.Q_n[:, :, i + 1] = self.Q_n[:, :, i]

            # Stops episode if terminal states is reached or tmax steps passed
            while (not reached_terminal) and t < tmax:

                # Take action according to epsilon greedy exploration policy
                action = self._choose_action_explr(state, eps, i+1)
                occurrences[state, action] += 1
                old_state = state
                state, reward, reached_terminal = self.env.step(state, action)

                reward_i += reward
                disct_reward_i += reward*weight
                weight = self.env.gamma*weight

                # Computes temporal difference (Q-Learning)
                td = self._compute_td(old_state, state, action, reward, i+1)

                # Updates the Q-function
                self.Q_n[old_state, action, i+1] = self.Q_n[old_state, action, i+1] + self.alpha[old_state, action](occurrences[old_state, action])*td
                t += 1

            # Saves the episode reward
            self.reward_n[i+1] = reward_i
            self.disct_reward_n[i+1] = disct_reward_i

        # Extract the greedy policy out of the Q-functions
        self._compute_greedy_policy(n)

    def _compute_greedy_policy(self, n):
        """
        :return: Computes greedy policy and value function out of the Q-functions
        """
        self.value_n = np.zeros((self.env.n_states, n+1))
        self.policy_n = np.zeros((self.env.n_states, n+1))
        for i in range(n+1):
            for state in range(self.env.n_states):
                # Exploiting relationship between value function and Q-function
                self.value_n[state, i] = self.Q_n[state, :, i].max()
                self.policy_n[state, i] = self.Q_n[state, :, i].argmax()

        self.policy = self.policy_n[:, - 1]

    def _compute_td(self, old_state, state, action, reward, i):
        """
        :return: Computes temporal difference Q-learning
        """
        td = reward - self.Q_n[old_state, action, i] + self.env.gamma * self.Q_n[state, :, i].max()
        return td

    def _choose_action_explr(self, state, eps, i):
        """
        :return: chooses next step action with an epsilon exploration policy
        """
        throw = random.random()
        available_actions = self.env.state_actions[state]

        if throw > eps:
            # exploitation: greedy action with probability 1-epsilon
            action = self.Q_n[state, :, i].argmax()
        else:
            # exploration with probability epsilon
            action = random.choice(available_actions)

        return action

    def plot_value_error(self, value_infty, label = '', ax = None):
        """
        :return: plot value function estimation error in terms of number of iterations
        """
        data_plot = np.abs(self.value_n - value_infty.reshape((-1, 1))).max(axis = 0)
        yline = 0
        title = r'$||v^* - v^{\pi_n}||_{\infty}$ in terms of number of iterations'
        ylabel = r'$||v^* - v^{\pi_n}||_{\infty}$'
        xlabel = 'Number of iterations n'
        self.plot_standard(data_plot, label, yline, title, ylabel, xlabel, ax)

    def plot_reward(self, discounted = False, yline = None, label = '', ax = None):
        """
        :return: Computes running average of reward of each episode
        """

        reward = self.reward_n if discounted == False else self.disct_reward_n
        data_plot = np.cumsum(reward) / np.arange(1, len(reward) +1 , 1)
        title = 'Empirical average of discounted reward of each episode' if discounted else 'Empirical average of reward of each episode'
        ylabel = 'Empirical average'
        xlabel = 'Number of iterations n'

        self.plot_standard(data_plot, label, yline, title, ylabel, xlabel, ax)

    @staticmethod
    def plot_standard(data_plot, label, yline, title, ylabel, xlabel, ax=None):
        if ax == None:
            f, ax = plt.subplots(1, figsize=(15, 8))

        ax.plot(np.arange(1, len(data_plot) + 1, 1), data_plot, label=label)
        ax.legend()
        if yline:
            ax.axhline(y=yline, color='r', linestyle='--')

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)



