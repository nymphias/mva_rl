import numpy as np
import matplotlib.pyplot as plt
import arms

class UCB(object):
    def __init__(self, T, MAB, ro):
        self.T = T
        self.MAB = MAB
        self.ro = ro

        self.n_bandits = len(self.MAB)
        self.t = None

        self.arm_scores = None
        self.est_means = None
        self.arm_pulls = None

        self.history_scores = None
        self.history_est_means = None
        self.history_reward = None
        self.history_pulls = None

    def initialize_UCB(self):
        self.arm_scores = np.zeros(self.n_bandits)
        self.arm_pulls = np.ones(self.n_bandits)
        self.est_means = np.array([float(bandit.sample()[0]) for bandit in self.MAB])
        self.t = self.n_bandits

        self.history_reward = np.zeros(self.T)
        self.history_pulls = -np.ones(self.T, dtype = int)
        self.history_reward[:self.n_bandits] = self.est_means
        self.history_pulls[:self.n_bandits] = np.arange(0, self.n_bandits, 1, dtype = int)

        self.history_scores = np.zeros((self.T, self.n_bandits))
        self.history_est_means = np.zeros((self.T, self.n_bandits))

    def sample(self, arm):
        if isinstance(arm, arms.ArmBernoulli):
            reward = arm.sample()

        else:
            reward_obs = arm.sample()
            bernoulli_draw = arms.ArmBernoulli(reward_obs)
            reward = bernoulli_draw.sample()

        return reward

    def compute_pull_per_arm(self):
        pull_track = self.history_pulls[:self.t].copy()
        pull_per_arm = np.bincount(pull_track)
        return pull_per_arm

    def compute_cum_reward_per_arm(self):
        pull_track = self.history_pulls[:self.t].copy()
        rew_track = self.history_reward[:self.t].copy()
        cum_reward_per_arm = np.array([rew_track[np.where(pull_track == bandit)[0]].sum() for bandit in
                  range(self.n_bandits)])
        return cum_reward_per_arm

    def compute_arm_score_UCB1(self):
        pull_per_arm = self.compute_pull_per_arm()
        self.arm_scores = self.est_means + self.ro * np.sqrt(0.5 * np.log(self.t - 1) / pull_per_arm)
        self.history_scores[self.t] = self.arm_scores

        arm_id = np.argmax(self.arm_scores)
        return arm_id

    def compute_arm_score_TS(self):
        pull_per_arm = self.compute_pull_per_arm()
        cum_reward_per_arm = self.compute_cum_reward_per_arm()
        self.arm_scores = np.array([np.random.beta(cum_reward_per_arm[bandit]+1, pull_per_arm[bandit]-cum_reward_per_arm[bandit]+1) for bandit in range(self.n_bandits)])
        self.history_scores[self.t] = self.arm_scores

        arm_id = np.argmax(self.arm_scores)
        return arm_id

    def compute_arm_score_naive(self):

        self.arm_scores = self.est_means
        self.history_scores[self.t] = self.arm_scores

        arm_id = np.argmax(self.arm_scores)
        return arm_id

    def run(self, mode = 'UCB1'):
        self.initialize_UCB()

        while self.t < self.T:
            arm_id = getattr(self, 'compute_arm_score_'+mode)()
            reward = self.sample(self.MAB[arm_id])

            self.history_pulls[self.t] = arm_id
            self.history_reward[self.t] = reward

            # recursive update of reward average estimate
            n_pull = len(self.history_pulls[self.history_pulls == arm_id])
            self.est_means[arm_id] = 1/n_pull * ((n_pull-1)*self.est_means[arm_id] + reward)
            self.history_est_means[self.t] = self.est_means

            self.t += 1
        return(self.history_reward, self.history_pulls)

    def plot_est_means(self, ylines, ax = None, strat = 'UCB1'):
        data_plots = [self.history_est_means[:,bandit] for bandit in range(self.n_bandits)]
        labels = [f'bandit_{i}' for i in range(self.n_bandits)]
        ylines = ylines
        title = f'Estimated average reward for strat: {strat}'
        ylabel = 'estimated average reward'
        xlabel = 't'
        self.plot_standard(data_plots, labels, ylines, title, ylabel, xlabel, ax=ax)

    def plot_scores(self, ax = None, strat = 'UCB1'):
        data_plots = [self.history_scores[:,bandit] for bandit in range(self.n_bandits)]
        labels = [f'bandit_{i}' for i in range(self.n_bandits)]
        yline = None
        title = f'Estimation of each bandit score for strat: {strat}'
        ylabel = 'score'
        xlabel = 't'
        self.plot_standard(data_plots, labels, yline, title, ylabel, xlabel, ax=ax)

    @staticmethod
    def plot_standard(data_plots, labels, ylines, title, ylabel, xlabel, ax=None):
        if ax == None:
            f, ax = plt.subplots(1, figsize=(15, 8))

        for i, data_plot in enumerate(data_plots):

            ax.plot(np.arange(1, len(data_plot) + 1, 1), data_plot, label=labels[i])

        ax.legend()
        if ylines is not None:
            for yline in ylines:
                ax.axhline(y=yline, color='r', linestyle='--')

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)