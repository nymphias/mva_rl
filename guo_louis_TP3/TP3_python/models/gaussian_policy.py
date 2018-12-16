import models
import numpy as np

class gaussian_policy(models.policy):
    def __init__(self, theta, sigma = 0.4):
        super().__init__(theta)
        self.sigma = sigma

    def draw(self, state):
        return np.random.normal(self.theta*state, self.sigma)

    def compute_traj_local_weight(self, action, state):
        mu = self.theta*state
        return ((action - mu)*state)/(self.sigma**2)

