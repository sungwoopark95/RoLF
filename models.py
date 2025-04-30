import numpy as np
from util import *
from abc import ABC, abstractmethod
from calculate_alpha import *
import scipy
from sklearn.linear_model import Lasso, LinearRegression
import statsmodels.api as sm
from typing import Callable

#############################################################################
############################ Multi-Armed Bandits ############################
#############################################################################
class MAB(ABC):
    @abstractmethod
    def choose(self): pass
    
    @abstractmethod
    def update(self, a, r): pass


class eGreedyMAB(MAB):
    def __init__(self, arms:int, epsilon:float, alpha:float=1.0, initial:float=0):
        self.arms = arms
        self.epsilon = epsilon
        self.alpha = alpha
        self.initial = initial
        self.counts = np.zeros(self.arms)
        self.values = np.zeros(self.arms) + self.initial
        self.t = 0

    def choose(self):
        self.t += 1
        # print(f"Round : {self.t}, Epsilon: {self.epsilon}")
        if np.random.random() < self.epsilon:
            return np.random.choice(self.arms)
        else:
            argmaxes, = np.where(self.values == np.max(self.values))
            return np.random.choice(argmaxes)
    
    def update(self, a, r):
        ## count update
        self.counts[a] += 1

        ## value update
        value = self.values[a]
        n = self.counts[a]
        new_value = (((n-1)/n) * value) + ((1/n) * r)
        self.values[a] = new_value

        ## epsilon update
        self.epsilon *= self.alpha


class ETC(MAB):
    ## Explore-then-Commit
    def __init__(self, arms:int, explore:int, horizon:int, initial:float=0):
        assert explore * arms <= horizon, "Explore must be less than or equal to horizon"
        self.explore = explore
        self.arms = arms
        self.initial = initial
        self.counts = np.zeros(self.arms)
        self.values = np.zeros(self.arms) + self.initial
        self.t = 0
    
    def choose(self):
        ## Exploration Step
        self.t += 1
        if (self.t-1) <= self.explore * self.arms:
            return (self.t-1) % self.arms

        ## Exploitation Step
        argmaxes, = np.where(self.values == np.max(self.values))
        return np.random.choice(argmaxes)

    def update(self, a, r):
        ## count update
        self.counts[a] += 1

        ## value update
        value = self.values[a]
        n = self.counts[a]
        new_value = (((n-1)/n) * value) + ((1/n) * r)
        self.values[a] = new_value


class UCBNaive(MAB):
    def __init__(self, n_arms:int, sigma:float=0.1, alpha:float=0.1, delta:float=0.1):
        self.n_arms = n_arms
        self.alpha = alpha
        self.delta = delta
        self.sigma = sigma
        self.counts = np.zeros(self.n_arms)
        self.qs = np.zeros(self.n_arms)
        self.ucbs = np.array([np.iinfo(np.int32).max for _ in range(self.n_arms)])
        self.t = 0
    
    def choose(self):
        self.t += 1
        returns = self.qs + self.ucbs
        argmaxes = np.where(returns == np.max(returns))[0]
        return np.random.choice(argmaxes)
    
    def update(self, a:int, r:float):
        """
        a: index of the chosen arm
        r: reward of the chosen arm
        """
        ## count update
        self.counts[a] += 1
        
        ## q update
        value = self.qs[a]
        n = self.counts[a]
        new_value = (((n-1)/n)*value) + ((1/n)*r)
        self.qs[a] = new_value
        
        ## ucb update
        inside = 2 * (self.sigma ** 2) * np.log(self.t/self.delta)
        self.ucbs[a] = self.alpha * np.sqrt(inside)


class UCBDelta(UCBNaive):
    def __init__(self, n_arms:int, delta:float):
        # set default values for sigma and alpha
        self.n_arms = n_arms
        self.delta = delta
        super().__init__(self.n_arms, delta=self.delta)
    
    def update(self, a:int, r:float):
        """
        a: index of the chosen arm
        r: reward of the chosen arm
        """
        ## count update
        self.counts[a] += 1
        
        ## q update
        value = self.qs[a]
        n = self.counts[a]
        new_value = (((n-1)/n)*value) + ((1/n)*r)
        self.qs[a] = new_value
        
        ## ucb update
        numerator = 2 * np.log(1/self.delta)
        self.ucbs[a] = np.sqrt(numerator / self.counts[a])


class UCBAsymptotic(UCBNaive):
    def __init__(self, arms:int):
        self.arms = arms
        super().__init__(self.n_arms)

    def update(self, a, r):
        ## count update
        self.counts[a] += 1
    
        ## q update
        value = self.qs[a]
        n = self.counts[a]
        new_value = (((n-1)/n)*value) + ((1/n)*r)
        self.qs[a] = new_value
    
        ## ucb update
        ft = 1 + (self.t * (np.log(self.t)**2))
        numerator = 2 * np.log(ft)
        self.ucbs[a] = np.sqrt(numerator / self.counts[a])


class UCBMOSS(UCBNaive):
    def __init__(self, arms:int, horizon:int):
        self.arms = arms
        self.horizon = horizon
        super().__init__(self.n_arms)
    
    def update(self, a, r):
        ## count update
        self.counts[a] += 1
        
        ## q update
        value = self.qs[a]
        n = self.counts[a]
        new_value = (((n-1)/n)*value) + ((1/n)*r)
        self.qs[a] = new_value
        
        ## ucb update
        left = 4 / n
        right = np.log(np.maximum(1, (self.horizon / (self.n_arms*n))))
        self.ucbs[a] = np.sqrt(left * right)


class ThompsonSampling(MAB):
    def __init__(self, arms:int, distribution:str):
        self.arms = arms
        assert distribution.lower() in ["bernoulli", "gaussian"], "Distribution must be either Bernoulli or Gaussian"
        self.distribution = distribution
        if distribution.lower() == "bernoulli":
            self.alphas = np.ones(self.arms)
            self.betas = np.ones(self.arms)
        elif distribution.lower() == "gaussian":
            self.mus = np.zeros(self.arms)
            self.sigmas = np.ones(self.arms)
        self.counts = np.zeros(shape=self.arms)
        self.qs = np.zeros(shape=self.arms)

    def choose(self):
        if self.distribution.lower() == "bernoulli":
            thetas = np.array([np.random.beta(a=alpha, b=beta) for (alpha, beta) in zip(self.alphas, self.betas)])
        elif self.distribution.lower() == "gaussian":
            thetas = np.array([np.random.normal(loc=mu, scale=var) for (mu, var) in zip(self.mus, self.sigmas)])
        argmaxes, = np.where(thetas == np.max(thetas))
        return np.random.choice(argmaxes)

    def update(self, a, r):
        ## count update
        self.counts[a] += 1
        
        ## q update
        value = self.qs[a]
        n = self.counts[a]
        new_value = (((n-1)/n)*value) + ((1/n)*r)
        self.qs[a] = new_value
        
        ## parameter update
        if self.distribution.lower() == "bernoulli":
            self.alphas[a] += r
            self.betas[a] += (1-r)
        elif self.distribution.lower() == "gaussian":
            self.mus[a] = new_value
            self.sigmas[a] = np.sqrt(1/n)


#############################################################################
############################ Contextual Bandits #############################
#############################################################################
class ContextualBandit(ABC):
    @abstractmethod
    def choose(self, x): pass
    
    @abstractmethod
    def update(self, x, r): pass


class LinUCB(ContextualBandit):
    def __init__(self, d:int, lbda:float, delta:float) -> None:
        self.d = d
        self.xty = np.zeros(d)
        self.Vinv = (1 / lbda) * np.identity(d)
        self.theta_hat = np.zeros(d)
        self.delta = delta
        self.t = 0
        
    def choose(self, x:np.ndarray) -> int:
        # x: action set at each round (N, d)
        self.t += 1
        
        ## compute the ridge estimator
        self.theta_hat = self.Vinv @ self.xty
        
        ## compute the ucb scores for each arm
        alpha = linucb_alpha(delta=self.delta) * np.sqrt(np.log(self.t))
        expected = x @ self.theta_hat # (N, ) theta_T @ x_t
        width = np.sqrt(np.einsum("Ni, ij, Nj -> N", x, self.Vinv, x)) # (N, ) widths
        ucb_scores = expected + (alpha * width) # (N, ) ucb score
        
        ## chose the argmax the ucb score
        maximum = np.max(ucb_scores)
        argmax, = np.where(ucb_scores == maximum)
        self.chosen_action = np.random.choice(argmax)
        return self.chosen_action
    
    def update(self, x:np.ndarray, r:float) -> None:
        # x: context of the chosen action (d, )
        chosen_context = x[self.chosen_action, :]
        self.Vinv = shermanMorrison(self.Vinv, chosen_context)
        self.xty += (r * chosen_context)
        
    def __get_param(self):
        return {"param": self.theta_hat}


class LinTS(ContextualBandit):
    def __init__(self, d:int, lbda:float, horizon:int, reward_std:float, delta:float) -> None:
        self.d = d
        self.Binv = (1 / lbda) * np.identity(d)
        self.xty = np.zeros(d)
        self.theta_hat = np.zeros(d)
        self.horizon = horizon
        self.reward_std = reward_std
        self.delta = delta
        self.t = 0
    
    def choose(self, x:np.ndarray) -> int:
        # x: action set at each round (N, d)
        self.t += 1
        
        ## compute the ridge estimator
        self.theta_hat = self.Binv @ self.xty
        
        ## parameter sampling
        # self.alpha_ = self.alpha * np.sqrt(np.log(self.t))
        # alpha = lints_alpha(d=self.d, horizon=self.horizon, reward_std=self.reward_std, delta=self.delta) * np.sqrt(np.log(self.t))
        alpha = lints_alpha(d=self.d, reward_std=self.reward_std, delta=self.delta)
        tilde_theta = np.random.multivariate_normal(mean=self.theta_hat, cov=(alpha**2) * self.Binv)  # (d, ) random matrix
        
        ## compute estimates and choose the argmax
        expected = x @ tilde_theta  # (N, ) vector
        maximum = np.max(expected)
        argmax, = np.where(expected == maximum)
        self.chosen_action = np.random.choice(argmax)
        return self.chosen_action
    
    def update(self, x:np.ndarray, r:float) -> None:
        # x: (K, d)
        # r: reward seen (scalar)
        chosen_context = x[self.chosen_action, :]
        self.Binv = shermanMorrison(self.Binv, chosen_context)
        self.xty += (r * chosen_context)
    
    def __get_param(self):
        return {"param": self.theta_hat}
    

class RoLFLasso(ContextualBandit):
    def __init__(self, d:int, arms:int, p:float, delta:float, sigma:float, random_state:int, explore:bool=False, init_explore:int=0):
        self.t = 0
        self.d = d
        self.K = arms
        self.mu_hat = np.zeros(self.K)      # main estimator
        self.mu_check = np.zeros(self.K)    # imputation estimator
        self.impute_prev = np.zeros(self.K)
        self.main_prev = np.zeros(self.K)
        self.sigma = sigma                  # variance of noise
        self.p = p                          # hyperparameter for action sampling
        self.delta = delta                  # confidence parameter
        self.action_history = []            # history of chosen actions up to the current round
        self.reward_history = []            # history of observed rewards up to the current round
        self.matching = dict()              # history of rounds that the pseudo action and the chosen action matched
        self.random_state = random_state
        self.explore = explore
        self.init_explore = init_explore

    def choose(self, x: np.ndarray):
        # x : (K, d) augmented feature matrix where each row denotes the augmented features
        self.t += 1

        ## compute the \hat{a}_t
        if self.explore:
            if self.t > self.init_explore:
                decision_rule = x @ self.mu_hat
                # print(f"Decision rule : {decision_rule}")
                a_hat = np.argmax(decision_rule)
            else:
                a_hat = np.random.choice(np.arange(self.K))
        else:
            decision_rule = x @ self.mu_hat
            # print(f"Decision rule : {decision_rule}")
            a_hat = np.argmax(decision_rule)

        self.a_hat = a_hat

        ## sampling actions
        pseudo_action = -1
        chosen_action = -2
        count = 0
        max_iter = int(np.log((self.t + 1) ** 2 / self.delta) / np.log(1 / self.p))
        pseudo_dist = np.array([(1 - self.p) / (self.K - 1)] * self.K, dtype=float)
        pseudo_dist[a_hat] = self.p
        chosen_dist = np.array([(1 / np.sqrt(self.t)) / (self.K - 1)] * self.K, dtype=float)
        chosen_dist[a_hat] = 1 - (1 / np.sqrt(self.t))

        np.random.seed(self.random_state + self.t)
        while (pseudo_action != chosen_action) and (count <= max_iter):
            ## Sample the pseudo action
            pseudo_action = np.random.choice([i for i in range(self.K)], size=1, replace=False, p=pseudo_dist).item()
            ## Sample the chosen action
            chosen_action = np.random.choice([i for i in range(self.K)], size=1, replace=False, p=chosen_dist).item()
            count += 1

        self.action_history.append(chosen_action) # add to the history
        self.pseudo_action = pseudo_action
        self.chosen_action = chosen_action
        return chosen_action

    def update(self, x: np.ndarray, r: float):
        # x : (K, K) augmented feature matrix
        # r : reward of the chosen_action
        self.reward_history.append(r)

        # lam_impute = 2 * self.p * self.sigma * np.sqrt(2 * self.t * np.log(2 * self.K * (self.t ** 2) / self.delta))
        # lam_main = (1 + 2 / self.p) * self.sigma * np.sqrt(2 * self.t * np.log(2 * self.K * (self.t ** 2) / self.delta))

        # lam_impute = self.p * np.sqrt(np.log(self.t))
        # lam_main = self.p * np.sqrt(np.log(self.t))

        lam_impute = self.p
        lam_main = self.p

        # print(f"x : {x.shape}")
        gram = x.T @ x
        gram_sqrt = matrix_sqrt(gram) 

        if self.pseudo_action == self.chosen_action:
            ## compute the imputation estimator
            data_impute = x[self.action_history, :]  # (t, d) matrix
            target_impute = np.array(self.reward_history)
            # print(f"gram_sqrt : {gram_sqrt.shape}")
            # print(f"impute_prev : {self.impute_prev.shape}")
            mu_impute = scipy.optimize.minimize(self.__imputation_loss, 
                                                (gram_sqrt @ self.impute_prev), 
                                                args=(data_impute, target_impute, lam_impute),
                                                method="SLSQP", 
                                                options={'disp': False, "ftol":1e-6, "maxiter":10000}).x

            ## compute and update the pseudo rewards
            if self.matching:
                for key in self.matching:
                    matched, data, _, chosen, reward = self.matching[key]
                    if matched:
                        new_pseudo_rewards = data @ mu_impute
                        new_pseudo_rewards[chosen] += (1 / self.p) * (reward - (data[chosen, :] @ mu_impute))
                        # overwrite the value
                        self.matching[key] = (matched, data, new_pseudo_rewards, chosen, reward)

            ## compute the pseudo rewards for the current data
            pseudo_rewards = x @ mu_impute
            pseudo_rewards[self.chosen_action] += (1 / self.p) * (r - (x[self.chosen_action, :] @ mu_impute))
            self.matching[self.t] = ((self.pseudo_action == self.chosen_action), x, pseudo_rewards, self.chosen_action, r)

            ## compute the main estimator
            mu_main = scipy.optimize.minimize(self.__main_loss, 
                                              (gram_sqrt @ self.main_prev), 
                                              args=(lam_main, self.matching),
                                              method="SLSQP", 
                                              options={'disp': False, "ftol":1e-6, "maxiter":10000}).x

            ## update the mu_hat
            self.mu_hat = mu_main
            self.mu_check = mu_impute
        else:
            self.matching[self.t] = ((self.pseudo_action == self.chosen_action), None, None, None, None)

    def __imputation_loss(self, beta:np.ndarray, X:np.ndarray, y:np.ndarray, lam:float):
        residuals = (y - (X @ beta)) ** 2
        loss = np.sum(residuals, axis=0)
        l1_norm = vector_norm(beta, type="l1")
        return loss + (lam * l1_norm)

    # def __main_loss(self, beta:np.ndarray, lam:float, matching_history:dict):
    #     ## matching_history : dict[t] = (bool, X, y) - bool denotes whether the matching event occurred or not
    #     loss = 0
    #     for key in matching_history:
    #         matched, X, pseudo_rewards, _, _ = matching_history[key]
    #         if matched:
    #             residuals = (pseudo_rewards - (X @ beta)) ** 2
    #             interim_loss = np.sum(residuals, axis=0)
    #         else:
    #             interim_loss = 0
    #         loss += interim_loss
    #     l1_norm = vector_norm(beta, type="l1")
    #     return loss + (lam * l1_norm)

    def __main_loss(self, beta: np.ndarray, lam: float, matching_history: dict):
        # Extract matched keys and data
        matched_keys = [key for key, value in matching_history.items() if value[0]]  # Filter matched entries
        X_list = [matching_history[key][1] for key in matched_keys]  # List of X matrices
        pseudo_rewards_list = [matching_history[key][2] for key in matched_keys]  # List of pseudo_rewards

        # Compute residuals for matched keys
        residuals_list = [
            (pseudo_rewards - X @ beta) ** 2 for X, pseudo_rewards in zip(X_list, pseudo_rewards_list)
        ]

        # Sum all residuals efficiently
        residuals_sum = sum(np.sum(residuals, axis=0) for residuals in residuals_list)

        # L1 regularization
        l1_norm = np.sum(np.abs(beta))

        # Total loss
        return residuals_sum + lam * l1_norm

    def __get_param(self):
        return {"param": self.mu_hat, "impute": self.mu_check}

class RoLFRidge(ContextualBandit):
    def __init__(self, d:int, arms:int, p:float, delta:float, sigma:float, random_state:int, explore:bool=False, init_explore:int=0):
        self.t = 0
        self.d = d
        self.K = arms
        self.mu_hat = np.zeros(self.K)      # main estimator
        self.mu_check = np.zeros(self.K)    # imputation estimator
        self.sigma = sigma                  # variance of noise
        self.p = p                          # hyperparameter for action sampling
        self.delta = delta                  # confidence parameter
        self.matching = dict()              # history of rounds that the pseudo action and the chosen action matched
        self.Vinv_impute = self.p * np.identity(self.K)
        self.xty_impute = np.zeros(self.K)
        self.random_state = random_state
        self.explore = explore
        self.init_explore = init_explore

    def choose(self, x: np.ndarray):
        # x : (K, d) augmented feature matrix where each row denotes the augmented features
        self.t += 1

        ## compute the \hat{a}_t
        if self.explore:
            if self.t > self.init_explore:
                decision_rule = x @ self.mu_hat
                # print(f"Decision rule : {decision_rule}")
                a_hat = np.argmax(decision_rule)
            else:
                a_hat = np.random.choice(np.arange(self.K))
        else:
            decision_rule = x @ self.mu_hat
            # print(f"Decision rule : {decision_rule}")
            a_hat = np.argmax(decision_rule)

        self.a_hat = a_hat

        ## sampling actions
        pseudo_action = -1
        chosen_action = -2
        count = 0
        max_iter = int(np.log((self.t + 1) ** 2 / self.delta) / np.log(1 / self.p))
        pseudo_dist = np.array([(1 - self.p) / (self.K - 1)] * self.K, dtype=float)
        pseudo_dist[a_hat] = self.p
        chosen_dist = np.array([(1 / np.sqrt(self.t)) / (self.K - 1)] * self.K, dtype=float)
        chosen_dist[a_hat] = 1 - (1 / np.sqrt(self.t))

        np.random.seed(self.random_state + self.t)
        while (pseudo_action != chosen_action) and (count <= max_iter):
            ## Sample the pseudo action
            pseudo_action = np.random.choice([i for i in range(self.K)], size=1, replace=False, p=pseudo_dist).item()
            ## Sample the chosen action
            chosen_action = np.random.choice([i for i in range(self.K)], size=1, replace=False, p=chosen_dist).item()
            count += 1

        self.pseudo_action = pseudo_action
        self.chosen_action = chosen_action
        # print(f"Round: {self.t}, a_hat: {a_hat}, pseudo_action: {pseudo_action}, chosen_action: {chosen_action}, count: {count}")
        return chosen_action

    def update(self, x: np.ndarray, r: float):
        # x : (K, K) augmented feature matrix
        # r : reward of the chosen_action
        if self.pseudo_action == self.chosen_action:
            ## compute the imputation estimator based on history
            chosen_context = x[self.chosen_action, :]
            self.Vinv_impute = shermanMorrison(self.Vinv_impute, chosen_context)
            self.xty_impute += (r * chosen_context)
            mu_impute = self.Vinv_impute @ self.xty_impute

            ## compute and update the pseudo rewards
            if self.matching:
                for key in self.matching:
                    matched, data, _, chosen, reward = self.matching[key]
                    if matched:
                        new_pseudo_rewards = data @ mu_impute
                        new_pseudo_rewards[chosen] += (1 / self.p) * (reward - (data[chosen, :] @ mu_impute))
                        # overwrite the value
                        self.matching[key] = (matched, data, new_pseudo_rewards, chosen, reward)

            ## compute the pseudo rewards for the current data
            pseudo_rewards = x @ mu_impute
            pseudo_rewards[self.chosen_action] += (1 / self.p) * (r - (x[self.chosen_action, :] @ mu_impute))
            self.matching[self.t] = ((self.pseudo_action == self.chosen_action), x, pseudo_rewards, self.chosen_action, r)

            ## compute the main estimator
            mu_main = self.__main_estimation(self.matching, dimension=self.K)

            ## update the mu_hat
            self.mu_hat = mu_main
            self.mu_check = mu_impute
        else:
            self.matching[self.t] = ((self.pseudo_action == self.chosen_action), None, None, None, None)

    # def __main_estimation(self, matching_history:dict, dimension:int):
    #     ## matching_history : dict[t] = (bool, X, y) - bool denotes whether the matching event occurred or not
    #     inv = np.identity(dimension)
    #     score = np.zeros(dimension, dtype=float)
    #     for key in matching_history:
    #         matched, X, pseudo_rewards, _, _ = matching_history[key]
    #         if matched:
    #             # inverse matrix
    #             inv_init = np.zeros(shape=(dimension, dimension))
    #             for a in range(X.shape[0]):
    #                 inv_init += np.outer(X[a, :], X[a, :])
    #             inv += inv_init

    #             # score
    #             score_init = np.zeros(shape=dimension, dtype=float)
    #             for a in range(X.shape[0]):
    #                 score_init += pseudo_rewards[a] * X[a, :]
    #             score += score_init

    #     return scipy.linalg.inv(inv) @ score

    def __main_estimation(self, matching_history: dict, dimension: int):
        # Initialize inv and score
        inv = np.identity(dimension)
        score = np.zeros(dimension, dtype=float)

        # Filter matched entries
        matched_entries = [value for key, value in matching_history.items() if value[0]]

        # Process matched entries
        for _, X, pseudo_rewards, _, _ in matched_entries:
            # Update inv (outer products of rows in X)
            inv += X.T @ X

            # Update score (weighted sum of rows in X)
            score += X.T @ pseudo_rewards

        # Compute final estimation
        return scipy.linalg.inv(inv) @ score

    def __get_param(self):
        return {"param": self.mu_hat, "impute": self.mu_check}

class DRLassoBandit(ContextualBandit):
    def __init__(self, d:int, arms:int, lam1:float, lam2:float, zT:float, tr:bool):
        ## learning params
        self.d = d
        self.arms = arms
        self.lam1 = lam1
        self.lam2 = lam2
        self.tr = tr
        self.zT = zT

        ## initialization
        self.beta_prev = np.zeros(self.d)
        self.beta_hat = np.zeros(self.d)
        self.pi_t = 0
        self.x = []     # containing context history
        self.r = []     # containing reward history
        self.t = 0      # learning round

    def choose(self, x):
        ## x : (K, d) array - all contexts observed at t
        self.t += 1
        if self.t <= self.zT:
            # forced sampling
            self.action = np.random.choice(self.arms, replace=False)
            self.pi_t = 1 / self.arms
        else:
            # UCB
            expected_reward = x @ self.beta_hat     # (K, ) array
            lam1 = self.lam1 * np.sqrt((np.log(self.t) + np.log(self.d)) / self.t)
            lam1 = np.minimum(1, np.maximum(0, lam1))
            self.mt = np.random.choice([0, 1], p=[1-lam1, lam1])
            if self.mt == 1:
                self.action = np.random.choice(self.arms)
            else:
                self.action = np.argmax(expected_reward)

            self.pi_t = (lam1 / self.arms) + ((1-lam1) * (self.action == np.argmax(expected_reward)))

        bar_x = np.mean(x, axis=0)
        self.x.append(bar_x)
        self.rhat = x @ self.beta_hat

        return self.action

    def update(self, x, r):
        ## x : (K, d) array - context of the all actions in round t
        ## r : float - reward
        r_hat = np.mean(self.rhat) + ((r - (x[self.action]@self.beta_hat)) / (self.arms * self.pi_t))
        if self.tr:
            r_hat = np.minimum(3., np.maximum(-3., r_hat))
        self.r.append(r_hat)

        lam2 = self.lam2 * np.sqrt((np.log(self.t) + np.log(self.d)) / self.t)
        data, target = np.vstack(self.x), np.array(self.r)
        self.beta_hat = scipy.optimize.minimize(self.__lasso_loss, self.beta_prev, args=(data, target, lam2),
                                                method="SLSQP", options={'disp': False, "ftol":1e-6, "maxiter":30000}).x

    def __lasso_loss(self, beta:np.ndarray, X:np.ndarray, y:np.ndarray, lam:float):
        loss = np.sum((y - X @ beta) ** 2, axis=0)
        l1norm = np.sum(np.abs(beta))
        return loss + (lam * l1norm)
    
    def __get_param(self):
        return self.beta_hat
    

class LassoBandit(ContextualBandit):
    def __init__(self, arms:int, horizon:int, d:int, q:int, h:float, lam1:float, lam2:float):
        ## input params for algorithms
        self.q = q              # input param 1 - for forced-sampling
        self.h = h              # input param 2
        self.lam1 = lam1        # input param 3
        self.lam2 = lam2        # input param 4

        ## basic params for bandits
        self.arms = arms        # the number of arms
        self.horizon = horizon  # learning horizon
        self.d = d              # dimension of features
        self.t = 0              # learning round; t <= horizon
        self.n = 0              # sample size

        ## sets
        self.Tx = {i: [] for i in range(self.arms)}
        self.Sx = {i: [] for i in range(self.arms)}
        self.Tr = {i: [] for i in range(self.arms)}
        self.Sr = {i: [] for i in range(self.arms)}

        ## estmators
        self.beta_t = np.zeros((self.arms, d))   # forced-sample estimators
        self.beta_s = np.zeros((self.arms, d))   # all samples estimators
        self.lasso_t = Lasso(alpha=lam1)

    def choose(self, x:np.ndarray):
        ## x: (d, ) array - context vector of time t
        self.t += 1

        flag = (((2**self.n)-1) * self.arms * self.q) + 1
        if self.t == flag:
            self.set = np.arange(self.t, self.t + (self.q*self.arms))
            self.n += 1

        if self.t in self.set:
            ## if t is in T_i for any i
            ind = list(self.set).index(self.t)
            self.action = ind // self.q
            self.Tx[self.action].append(x)
        else:
            ## if indices is none
            expected_T = self.beta_t @ x
            max_expected_T = np.amax(expected_T)
            K_hat = np.argwhere(expected_T >= (max_expected_T - (self.h/2))).flatten()

            expected_S = self.beta_s @ x
            filtered_expected = expected_S[K_hat]
            argmax = np.argmax(filtered_expected)
            self.action = K_hat[argmax]

        self.Sx[self.action].append(x)        # append the context of the actually chosen action
        return self.action

    def update(self, r:float):
        if self.t in self.set:
            self.Tr[self.action].append(r)
            ## update beta_t using Lasso
            data_t, target_t = np.vstack(self.Tx[self.action]), np.array(self.Tr[self.action])
            # print(data_t.shape)
            beta_t = scipy.optimize.minimize(self.__lasso_loss, np.zeros(d), args=(data_t, target_t, self.lam1),
                                             method="SLSQP", options={'disp': False, "ftol":1e-6, "maxiter":10000}).x

            self.beta_t[self.action] = beta_t

        self.Sr[self.action].append(r)
        ## update beta_s using Lasso
        lam2_t = self.lam2 * np.sqrt(((np.log(self.t) + np.log(self.d)) / self.t))
        data_s, target_s = np.vstack(self.Sx[self.action]), np.array(self.Sr[self.action])
        # print(f"action : {self.action}, data : {data_s.shape}")
        beta_s = scipy.optimize.minimize(self.__lasso_loss, np.zeros(d), args=(data_s, target_s, lam2_t),
                                         method="SLSQP", options={'disp': False, "ftol":1e-6, "maxiter":10000}).x

        self.beta_s[self.action] = beta_s

    def __lasso_loss(self, beta:np.ndarray, X:np.ndarray, y:np.ndarray, lam:float):
        # print(X)
        # print(f"X : {X.shape}, beta : {beta.shape}")
        loss = np.sum((y - X @ beta) ** 2, axis=0)
        l1norm = np.sum(np.abs(beta))
        return (loss + (lam*l1norm))
