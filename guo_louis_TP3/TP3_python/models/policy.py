import numpy as np
class policy(object):
    def __init__(self, theta):
        self.theta = theta
        self.N = None
        self.T = None

    def compute_traj_reward(self, trajectories, gamma):
        discounts = np.array([gamma**t for t in np.arange(0, self.T)])
        traj_rewards = np.array([discounts * trajectories[i]["rewards"] for i in np.range(0,self.N)])
        return traj_rewards

    def compute_traj_weight(self, trajectories):
        weights = np.array([np.sum([self.compute_traj_local_weight(trajectories[i]["actions"][t], trajectories[i]["states"][t]) for t in range(self.T)]) for i in range(self.N)])
        return weights

    def compute_Jgrad(self, trajectories, gamma):
        self.N, self.T = len(trajectories), len(trajectories[0]['states'])
        # Trajectory rewards: size N
        traj_rewards = self.compute_traj_reward(trajectories, gamma)

        # Weights: size N
        weights = self.compute_traj_weight(trajectories)

        return np.mean(traj_rewards * weights)
