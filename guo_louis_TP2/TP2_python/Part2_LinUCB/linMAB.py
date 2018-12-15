import numpy as np
import matplotlib.pyplot as plt
import tqdm
import random

class linMAB(object):
    def __init__(self, env, alpha = 2, lambd = 1, T = 6000, epsilon = 0.5):
        self.env = env
        self._alpha = alpha
        self._lambda = lambd
        self._epsilon = epsilon

        self.n_a = env.n_actions
        self.d = env.n_features

        self.T = T

        # Metrics:
        self.norm_dist = np.zeros(self.T)
        self.regret = np.zeros(self.T)

        #
        self.A = None
        self.b = None

    def initialize_linMAB(self):
        self.A = self._lambda * np.identity(self.d)
        self.b = np.zeros(self.d)

    def compute_best_arm_UCB(self, theta_hat):
        exploitation = self.env.features @ theta_hat
        exploration = self._alpha * np.sqrt(np.diagonal(self.env.features @ np.linalg.inv(self.A) @ self.env.features.T))

        return np.argmax(exploitation + exploration)

    def compute_arm_greedy_eps(self, theta_hat):
        draw = random.random()
        if draw > self._epsilon:
            # Greedy choice
            return np.argmax(self.env.features @ theta_hat)
        else:
            return np.random.randint(self.n_a)

    def run(self, mode = 'linUCB'):
        self.initialize_linMAB()
        for t in range(self.T):
            # Estimation of reward coefficient using regularized least squares
            theta_hat = np.linalg.inv(self.A) @ (self.b)

            if mode == 'linUCB':
                # Choose best arm with UCB criterion
                a_t = self.compute_best_arm_UCB(theta_hat)
            elif mode == 'random':
                # Choose arm randomly (uniformly)
                a_t = np.random.randint(self.n_a)
            elif mode == 'eps_greedy':
                a_t = self.compute_arm_greedy_eps(theta_hat)

            # Get the observed reward
            r_t = self.env.reward(a_t)

            # Adding new observation feature to A and b
            new_feature = self.env.features[a_t, :].reshape((-1, 1))
            self.A += new_feature @ new_feature.T
            self.b += r_t * new_feature.flatten()

            # Computes metric
            self.regret[t] = self.env.best_arm_reward() - r_t
            self.norm_dist[t] = np.linalg.norm(self.env.real_theta - theta_hat, 2)

        return self.regret, self.norm_dist

    @staticmethod
    def plot_standard(data_plots, labels, ylines, title, ylabel, xlabel, ax=None):
        c = [(h/270, 0.5,0.5) for h in np.linspace(0, 270, len(data_plots))]

        if ax == None:
            f, ax = plt.subplots(1, figsize=(15, 8))

        for i, data_plot in enumerate(data_plots):

            ax.plot(np.arange(1, len(data_plot) + 1, 1), data_plot, label=labels[i], color = c[i])

        ax.legend()
        if ylines is not None:
            for yline in ylines:
                ax.axhline(y=yline, color='r', linestyle='--')

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)