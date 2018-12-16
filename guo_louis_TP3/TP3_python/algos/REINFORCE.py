import utils as tls
import models

import numpy as np
from tqdm import tqdm

class REINFORCE(object):
    def __init__(self, env, policy_model, gamma, update_rate, n_steps, N, T, theta_0, *args, **kwargs):
        # Environment
        self.env = env
        self.policy_class = getattr(models, policy_model)
        self.policy = self.policy_class(theta = self.theta, *args, **kwargs)
        self.theta = theta_0
        self.gamma = gamma

        # Gradient update rule
        self.update_rate = update_rate
        self.n_steps = n_steps

        # Trajectory parameters:
        self.N = N
        self.T = T

        # Records:
        self.thetas = []
        self.Js = []

    def run(self, record_J = False):
        self.thetas.append(self.theta)
        for _ in tqdm(np.arange(0, self.n_steps)):
            self.policy.set_theta(self.theta)
            trajectories = tls.collect_episodes(self.env, policy = self.policy, horizon = self.T, n_episodes=self.N)

            if record_J:
                J = tls.estimate_performance(paths = trajectories)
                self.Js.append(J)

            # Compute gradient ascent update
            grad = self.policy.compute_Jgrad(trajectories, gamma = self.gamma)
            ascent = self.update_rate.update(grad)
            self.theta = self.theta + ascent
            self.thetas.append(self.theta)
