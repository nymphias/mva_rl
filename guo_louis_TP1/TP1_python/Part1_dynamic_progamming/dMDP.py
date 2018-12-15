import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class dMDP(object):
    # implements discrete Markov Decision Process model

    # Discounted Infinite Horizon framework
    def __init__(self, n_state, n_action, dyn, reward, gamma, tol, solver):
        """
        :param n_state: number of states
        :param n_action: number of actions
        :param dyn: dynamic of MDP #(n_state, n_state, n_action) giving transition probabilities
        :param reward: reward #(n_state, n_actions)
        :param gamma: discount factor

        :param tol: stopping criterion st we obtain a tol-optimal policy
        :param solver: solver mode (Policy Iteration or Value Iteration) to find optimal policy

        Additional attributes:
             n_iter_ : number of iterations ran
             eps : stopping criterion on the distance of two consecutive iterated value function
             valuef_init (resp. policy_init): value function (resp. policy) at initialization #(n_state)
             valuef_ (resp. policy_): current value function (resp. policy) during iteration #(n_state)
             valuef_opt_ (resp. policy_opt_): estimation of optimal value function (resp. policy) after iterations

             valuef_history: history of value functions #(N_iterations, n_state)
             error_history_ : history of distance between optimal value and current value function (L_infty) #(N_iterations)
        """
        self.n_state = n_state
        self.n_action = n_action
        self.dyn = dyn

        assert (dyn.shape == (n_state, n_state, n_action))

        self.reward = reward
        self.gamma = gamma

        self.n_iter = 0
        self.tol = tol
        self.eps = self.tol/2 * (1-self.gamma)/self.gamma
        self.solver = solver

        self.valuef_init_ = None
        self.valuef_ = None
        self.valuef_opt_ = None

        self.policy_init_ = None
        self.policy_ = None
        self.policy_opt_ = None

        self.valuef_history = []
        self.error_history_ = None

    def _apply_opt_boperator(self):
        """
        :return: Apply optimal Bellman operator on current iterated value function and updates it
        """
        self.valuef_ = np.array([np.max(self.reward[x,:] + self.gamma * self.valuef_ @ self.dyn[x,:,:]) for x in range(self.n_state)])

    def _compute_greedy_policy_from_value(self):
        """
        :return: Computes greedy policy from value function
        """
        self.policy_ = np.array([np.argmax(self.reward[x, :] + self.gamma * self.valuef_ @ self.dyn[x, :, :]) for x in range(self.n_state)])

    @staticmethod
    def _compute_infty_norm(arr):
        """
        :return: Computes L_infty norm of arr
        """
        return np.max(np.abs(arr))

    def _evaluate_policy(self):
        """
        :return: Evaluates policy by direct computation of value function (inverting Bellman equation)
        """
        r_pi = np.array([self.reward[x, self.policy_[x]] for x in range(self.n_state)])
        p_pi = np.asarray([self.dyn[x, :, self.policy_[x]] for x in range(self.n_state)])

        res = np.linalg.inv(np.identity(self.n_state) - self.gamma * p_pi) @ r_pi
        return res

    def _initiate_value(self, value):
        """
        :return: Sets initial value function to value
        """
        self.valuef_init = value
        self.valuef_ = value

    def _solveVI(self, value_init):
        """
        :return: Use value iteration to converge to optimal value function
        """

        # initiates value function with user-defined initialization
        self._initiate_value(value_init)

        self.n_iter += 1
        previous_valuef_ = self.valuef_
        self.valuef_history.append(self.valuef_)
        self._apply_opt_boperator()

        # Iterating while two consecutive value functions are distant of more than eps
        while self._compute_infty_norm(self.valuef_ - previous_valuef_) > self.eps:

            # Each iteration, we apply optimal Bellman operator
            self.n_iter +=1
            previous_valuef_ = self.valuef_
            self._apply_opt_boperator()
            self.valuef_history.append(self.valuef_)

        # Policy extraction (greedy policy computed from value function)
        self._compute_greedy_policy_from_value()

        self.policy_opt_ = self.policy_
        # Computes back the value function of the extracted policy
        self.valuef_opt_ = self._evaluate_policy()
        self._compute_valuef_history()

    def _initiate_policy(self, policy):
        """
        :return: Sets initial policy to policy
        """
        self.policy_init = policy
        self.policy_ = policy

    def _solvePI(self, policy_init):
        """
        :return: Use policy iteration to find exact optimal policy
        """

        # initiates policy with user-defined initialization
        self._initiate_policy(policy_init)

        res = True
        # Policy evaluation
        previous_valuef_ = -np.inf*np.ones((self.n_state))

        # Iterates till two consecutive policy are the same
        while res:
            self.n_iter +=1
            # Policy evaluation
            self.valuef_ = self._evaluate_policy()
            res = (self._compute_infty_norm(self.valuef_ - previous_valuef_) != 0)
            previous_valuef_ = self.valuef_
            self.valuef_history.append(self.valuef_)

            # Policy improvement
            self._compute_greedy_policy_from_value()

        self.policy_opt_ = self.policy_
        self.valuef_opt_ = self.valuef_
        self._compute_valuef_history()

    def _compute_valuef_history(self):
        """
        :return: Computes L_infty distance of value function to optimal value function
        """
        self.valuef_history = np.asarray(self.valuef_history)
        self.error_history_ = pd.DataFrame(self.valuef_history - self.valuef_opt_).apply(self._compute_infty_norm, axis = 1)

    def plot_error(self):
        """
        :return: Evolution of loss function
        """
        self.error_history_.index = np.arange(1, len(self.error_history_.index) +1, 1)
        self.error_history_.plot(marker = 'x', figsize = (15,8))
        plt.axhline(y = 0, color = 'r', linestyle = '--')
        # plt.title('L_inf distance between vk and v*')
        plt.title(r'$|| v_k - v^* ||_\infty$ in terms of nb of iterations')

        plt.ylabel(r'$|| v_k - v^* ||_\infty$')
        plt.xlabel('Number of iterations')
        plt.show()


    def solve(self, *args, **kwargs):
        """
        :return: solve Part1_dynamic_progamming problem with Value Iteration or Policy Iteration method
        """

        if self.solver == "VI":
            self._solveVI(*args, **kwargs)

        if self.solver == "PI":
            self._solvePI(*args, **kwargs)

