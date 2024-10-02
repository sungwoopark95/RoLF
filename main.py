from cfg import get_cfg
from models import *
from util import *

# MOTHER_PATH = "/home/sungwoopark/bandit-research/rolf"
MOTHER_PATH = "."

DIST_DICT = {
    "gaussian": "g",
    "uniform": "u"
}

AGENT_DICT = {
    "mab_ucb": r"UCB($\delta$)",
    "linucb": "LinUCB",
    "lints": "LinTS",
    "rolf_lasso": "RoLF-Lasso",
    "rolf_ridge": "RoLF-Ridge",
    "dr_lasso": "DRLasso"
}

cfg = get_cfg()

def run_trials(agent_type:str, trials:int, horizon:int, k:int, d:int, arms:int, noise_std:float, random_state:int, verbose:bool):
    exp_map = {
        "double": (2 * arms),
        "sqr": (arms ** 2),
        "K": arms,
        "triple": (3 * arms),
        "quad": (4 * arms)
    } 

    ## sample the observable features and orthogonal basis, then augment the feature factor
    # Z = feature_sampler(dimension=k, feat_dist=cfg.feat_dist, size=arms, disjoint=cfg.feat_disjoint, 
    #                     cov_dist=cfg.feat_cov_dist, bound=cfg.feat_feature_bound, bound_method=cfg.feat_bound_method, 
    #                     bound_type=cfg.feat_bound_type, uniform_rng=cfg.feat_uniform_rng, random_state=random_state_) # (K, k)
    np.random.seed(random_state)
    # rho_sq = cfg.rho_sq
    V = np.eye(arms)
    Z = np.random.multivariate_normal(mean=np.zeros(arms), cov=V, size=k)   # (k, K)
    X = Z[:d, :]    # (d, K)

    ## run and collect the regrets
    regret_container = np.zeros(trials, dtype=object)
    for trial in range(trials):
        if random_state is not None:
            random_state_ = random_state + (513 * trial)
        else:
            random_state_ = None

        if agent_type == "linucb":
            agent = LinUCB(d=d, lbda=cfg.p, delta=cfg.delta)
        elif agent_type == "lints":
            agent = LinTS(d=d, lbda=cfg.p, horizon=horizon, reward_std=noise_std, delta=cfg.delta)
        elif agent_type == "mab_ucb":
            agent = UCBDelta(n_arms=arms, delta=cfg.delta)
        elif agent_type == "rolf_lasso":
            if cfg.explore:
                agent = RoLFLasso(d=d, arms=arms, p=cfg.p, delta=cfg.delta, sigma=noise_std, random_state=random_state_, 
                                  explore=cfg.explore, init_explore=exp_map[cfg.init_explore])
            else:
                agent = RoLFLasso(d=d, arms=arms, p=cfg.p, delta=cfg.delta, sigma=noise_std, random_state=random_state_)                
        elif agent_type == "rolf_ridge":
            if cfg.explore:
                agent = RoLFRidge(d=d, arms=arms, p=cfg.p, delta=cfg.delta, sigma=noise_std, random_state=random_state_,
                                  explore=cfg.explore, init_explore=exp_map[cfg.init_explore])
            else:
                agent = RoLFRidge(d=d, arms=arms, p=cfg.p, delta=cfg.delta, sigma=noise_std, random_state=random_state_)
        elif agent_type == "dr_lasso":
            agent = DRLassoBandit(d=d, arms=arms, lam1=1., lam2=0.5, zT=10, tr=True)

        ## sample reward parameter after augmentation and compute the expected rewards
        reward_param = param_generator(dimension=k, distribution=cfg.param_dist, disjoint=cfg.param_disjoint, bound=cfg.param_bound, 
                                    bound_type=cfg.param_bound_type, uniform_rng=cfg.param_uniform_rng, random_state=random_state_)
        exp_rewards = Z.T @ reward_param # (K, ) vector

        if isinstance(agent, LinUCB) or isinstance(agent, LinTS) or isinstance(agent, DRLassoBandit):
            data = X.T  # (K, d)
        else:
            # (K, K-d) matrix and each column vector denotes the orthogonal basis if K > d
            # (K, K) matrix from singular value decomposition if d > K
            basis = orthogonal_complement_basis(X) 

            d, K = X.shape
            if d <= K:
                x_aug = np.hstack((X.T, basis)) # augmented into (K, K) matrix and each row vector denotes the augmented feature
                data = x_aug
            else:
                data = basis
        
        bounding(type="feature", v=data, bound=cfg.feat_feature_bound, method=cfg.feat_bound_method, norm_type="lsup")
        regrets = run(trial=trial, agent=agent, horizon=horizon, exp_rewards=exp_rewards, x=data, 
                      noise_dist=cfg.reward_dist, noise_std=noise_std, random_state=random_state_, verbose=verbose)
        regret_container[trial] = regrets
    return regret_container


def run(trial:int, agent:Union[MAB, ContextualBandit], horizon:int, exp_rewards:np.ndarray, 
        x:np.ndarray, noise_dist:str, noise_std:float, random_state:int, verbose:bool):
    # x: augmented feature if the agent is RoLF (K, K)
    regrets = np.zeros(horizon, dtype=float)

    if not verbose:
        bar = tqdm(range(horizon))
    else:
        bar = range(horizon)

    for t in bar:
        if random_state is not None:
            random_state_ = random_state + int(113 * t)
        else:
            random_state_ = None

        if t == 0:
            print(f"Number of actions : {x.shape[0]}\tReward range : [{np.amin(exp_rewards):.5f}, {np.amax(exp_rewards):.5f}]")
        
        ## compute the optimal action
        optimal_action = np.argmax(exp_rewards)
        optimal_reward = exp_rewards[optimal_action]

        ## choose the best action
        noise = subgaussian_noise(distribution=noise_dist, size=1, std=noise_std, random_state=random_state_)
        if isinstance(agent, ContextualBandit):
            chosen_action = agent.choose(x)
        else:
            chosen_action = agent.choose()
        chosen_reward = exp_rewards[chosen_action] + noise
        if verbose:
            try:
                print(f"SEED : {cfg.seed}, K : {cfg.arms}, Obs_dim : {cfg.dim}, Trial : {trial}, p : {cfg.p}, Agent : {agent.__class__.__name__}, Round : {t+1}, optimal : {optimal_action}, a_hat: {agent.a_hat}, pseudo : {agent.pseudo_action}, chosen : {agent.chosen_action}")
            except:
                print(f"SEED : {cfg.seed}, K : {cfg.arms}, Obs_dim : {cfg.dim}, Trial : {trial}, p : {cfg.p}, Agent : {agent.__class__.__name__}, Round : {t+1}, optimal : {optimal_action}, chosen : {chosen_action}")

        ## compute the regret
        regrets[t] = optimal_reward - exp_rewards[chosen_action]

        ## update the agent
        if isinstance(agent, ContextualBandit):
            agent.update(x=x, r=chosen_reward)
        else:
            agent.update(a=chosen_action, r=chosen_reward)

    return np.cumsum(regrets)


def show_result(regrets:dict, horizon:int, arms:int, figsize:tuple=(6, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['orange', 'blue', 'green', 'red', 'purple', 'black']
    period = horizon // 10
    
    # Plot the graph for each algorithm with error bars
    for color, (key, item) in zip(colors, regrets.items()):
        rounds = np.arange(horizon)
        mean = np.mean(item, axis=0)
        std = np.std(item, axis=0, ddof=1)
        
        # Display the line with markers and error bars periodically
        ax.errorbar(rounds[::period], mean[::period], yerr=std[::period], label=f"{key}", 
                    fmt='s', color=color, capsize=3, elinewidth=1)
        
        # Display the full line without periodic markers
        ax.plot(rounds, mean, color=color, linewidth=2)
    
    ax.grid(True)
    ax.set_xlabel(r"Round ($t$)")
    ax.set_ylabel("Cumulative Regret")
    ax.legend(fontsize=11)
    
    fig.tight_layout()  
    return fig


if __name__ == "__main__":
    ## hyper-parameters
    arms = cfg.arms # List[int]
    k = cfg.latent_dim
    d = cfg.dim
    T = cfg.horizon
    SEED = cfg.seed
    # AGENTS = ["rolf_lasso", "rolf_ridge", "dr_lasso", "linucb", "lints", "mab_ucb"]
    AGENTS = ["rolf_ridge", "rolf_lasso", "dr_lasso", "linucb", "lints", "mab_ucb"]

    RESULT_PATH = f"{MOTHER_PATH}/results/seed_{cfg.seed}_p_{cfg.p}_std_{cfg.reward_std}"
    FIGURE_PATH = f"{MOTHER_PATH}/figures/seed_{cfg.seed}_p_{cfg.p}_std_{cfg.reward_std}"
   
    regret_results = dict()
    for agent_type in AGENTS:
        regrets = run_trials(agent_type=agent_type, trials=cfg.trials, horizon=T, k=k, d=d, 
                             arms=arms, noise_std=cfg.reward_std, random_state=SEED, verbose=True)
        key = AGENT_DICT[agent_type]
        regret_results[key] = regrets
    
    fname = f"Seed_{SEED}_K_{arms}_d_{d}_T_{T}_p_{cfg.p}_delta_{cfg.delta}_explored_{cfg.init_explore}_param_{DIST_DICT[cfg.param_dist]}"
    fig = show_result(regrets=regret_results, horizon=T, arms=arms)
    save_plot(fig, path=FIGURE_PATH, fname=fname)