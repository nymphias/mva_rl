import steppers
import numpy as np

class adam_stepper(steppers.stepper):
    def __init__(self, alpha, betas, epsilon, m, v):
        self.alpha = alpha
        self.beta_1, self.beta_2 = betas[0], betas[1]
        self.epsilon = epsilon
        self.m = m
        self.v = v
        self.t = 0

    def update(self, gt):
        self.t += 1
        self.m = self.beta_1*self.m + (1-self.beta_1)*gt
        self.v = self.beta_2*self.v + (1-self.beta_2)*gt**2

        m_sized = self.m / (1-self.beta_1**self.t)
        v_sized = self.v / (1-self.beta_2**self.t)
        ascent = self.alpha * m_sized / (np.sqrt(v_sized) + self.epsilon)
        return ascent
