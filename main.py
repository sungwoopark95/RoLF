from cfg import get_cfg
from models import *
from util import *

MOTHER_PATH = "."

DIST_DICT = {
    "gaussian": "g",
    "uniform": "u"
}

AGENT_DICT = {
    "mab_ucb": r"UCB($\delta$)",
    "linucb": "LinUCB",
    "lints": "LinTS",
    "rolf_lasso": "RoLF-Lasso (Ours)",
    "rolf_ridge": "RoLF-Ridge (Ours)",
    "dr_lasso": "DRLasso"
}

cfg = get_cfg()

RESULT_PATH = f"{MOTHER_PATH}/results/{str(cfg.date)}/seed_{cfg.seed}_p_{cfg.p}_std_{cfg.reward_std}"
FIGURE_PATH = f"{MOTHER_PATH}/figures/{str(cfg.date)}/seed_{cfg.seed}_p_{cfg.p}_std_{cfg.reward_std}"
LOG_PATH = f"{MOTHER_PATH}/logs/{str(cfg.date)}/seed_{cfg.seed}_p_{cfg.p}_std_{cfg.reward_std}"


def feature_generator(case:int, 
                      d_z:int,
                      d:int,
                      K:int,
                      random_state:int):
    ## sample the true, observable, and unobservable features
    np.random.seed(random_state)
    d_u = d_z - d   # dimension of unobserved features
    assert case in [1, 2, 3], "There exists only Case 1, 2, and 3."
    if case == 1:
        ## Default case
        Z = np.random.multivariate_normal(mean=np.zeros(d_z), 
                                          cov=np.eye(d_z), 
                                          size=K).T   # (k, K)
        X = Z[:d, :]    # (d, K)


    # For two matrices A and B, 
    # if each row of A can be expressed as a linear combination of the rows of B, 
    # then R(A) ⊆ R(B)
    elif case == 2:
        ## R(U) ⊆ R(X)
        
        # First generate X
        X = np.random.multivariate_normal(mean=np.zeros(d), 
                                          cov=np.eye(d), 
                                          size=K).T # (d, K)
        
        # Generate a coefficient matrix C
        # C = np.random.multivariate_normal(mean=np.zeros(d), 
        #                                   cov=np.eye(d), 
        #                                   size=d_u).T # (d_u, d)
        C = np.random.uniform(low=-1/np.pi,
                              high=1/np.pi,
                              size=(d_u, d)) # (d_u, d)
        
        # Compute U as a multiplication between C and X
        U = C @ X # (d_u, K)
        Z = np.concatenate([X, U], axis=0) # (k, K)))
    
    elif case == 3:
        ## R(X) ⊆ R(U)
        # First generate U
        U = np.random.multivariate_normal(mean=np.zeros(d_u), 
                                          cov=np.eye(d_u), 
                                          size=K).T # (d_u, K)
        
        # Generate a coefficient matrix C
        # C = np.random.multivariate_normal(mean=np.zeros(d), 
        #                                   cov=np.eye(d), 
        #                                   size=d_u).T # (d, d_u)
        C = np.random.uniform(low=-1/np.pi, 
                              high=1/np.pi, 
                              size=(d, d_u)) # (d, d_u)
        
        # Compute U as a multiplication between C and X
        X = C @ U # (d, K)
        Z = np.concatenate([X, U], axis=0) # (k, K)))
    
    return Z, X


def run_trials(agent_type:str, 
               trials:int, 
               horizon:int, 
               k:int, 
               d:int, 
               arms:int, 
               noise_std:float, 
               case:int,
               random_state:int, 
               verbose:bool,
               fname:str):
    
    exp_map = {
        "double": (2 * arms),
        "sqr": (arms ** 2),
        "K": arms,
        "triple": (3 * arms),
        "quad": (4 * arms)
    } 
    
    ## run and collect the regrets
    regret_container = np.zeros(trials, dtype=object)
    for trial in range(trials):
        if random_state is not None:
            random_state_ = random_state + (513 * trial)
        else:
            random_state_ = None

        if agent_type == "linucb":
            agent = LinUCB(d=d, 
                           lbda=cfg.p, 
                           delta=cfg.delta)
            
        elif agent_type == "lints":
            agent = LinTS(d=d, 
                          lbda=cfg.p, 
                          horizon=horizon, 
                          reward_std=noise_std, 
                          delta=cfg.delta)
            
        elif agent_type == "mab_ucb":
            agent = UCBDelta(n_arms=arms, 
                             delta=cfg.delta)
            
        elif agent_type == "rolf_lasso":
            if cfg.explore:
                agent = RoLFLasso(d=d, 
                                  arms=arms, 
                                  p=cfg.p, 
                                  delta=cfg.delta, 
                                  sigma=noise_std, 
                                  random_state=random_state_, 
                                  explore=cfg.explore, 
                                  init_explore=exp_map[cfg.init_explore])
            else:
                agent = RoLFLasso(d=d, 
                                  arms=arms, 
                                  p=cfg.p, 
                                  delta=cfg.delta, 
                                  sigma=noise_std, 
                                  random_state=random_state_)
                
        elif agent_type == "rolf_ridge":
            if cfg.explore:
                agent = RoLFRidge(d=d, 
                                  arms=arms, 
                                  p=cfg.p, 
                                  delta=cfg.delta, 
                                  sigma=noise_std, 
                                  random_state=random_state_,
                                  explore=cfg.explore, 
                                  init_explore=exp_map[cfg.init_explore])
            else:
                agent = RoLFRidge(d=d, 
                                  arms=arms, 
                                  p=cfg.p, 
                                  delta=cfg.delta, 
                                  sigma=noise_std, 
                                  random_state=random_state_)
                
        elif agent_type == "dr_lasso":
            agent = DRLassoBandit(d=d, 
                                  arms=arms, 
                                  lam1=1., 
                                  lam2=0.5, 
                                  zT=10, 
                                  tr=True)
            
        ## sample features
        Z, X = feature_generator(case=case,
                                 d_z=k,
                                 d=d,
                                 K=arms,
                                 random_state=random_state_+1)

        ## sample reward parameter after augmentation and compute the expected rewards
        reward_param = param_generator(dimension=k, 
                                       distribution=cfg.param_dist, 
                                       disjoint=cfg.param_disjoint, 
                                       bound=cfg.param_bound, 
                                       bound_type=cfg.param_bound_type, 
                                       uniform_rng=cfg.param_uniform_rng, 
                                       random_state=random_state_)
        
        ## (K, ) vector with the maximum absolute value does not exceed 1
        exp_rewards = bounding(type="param", 
                               v=Z.T @ reward_param, 
                               bound=1., 
                               norm_type="lsup") 

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
        
        # print(f"Agent : {agent.__class__.__name__}\t data shape : {data.shape}")
        
        regrets = run(trial=trial, 
                      agent=agent, 
                      horizon=horizon, 
                      exp_rewards=exp_rewards, 
                      x=data, 
                      noise_dist=cfg.reward_dist, 
                      noise_std=noise_std, 
                      random_state=random_state_, 
                      verbose=verbose,
                      fname=fname)
        
        regret_container[trial] = regrets
    return regret_container


def run(trial:int, 
        agent:Union[MAB, ContextualBandit], 
        horizon:int, 
        exp_rewards:np.ndarray, 
        x:np.ndarray, 
        noise_dist:str, 
        noise_std:float, 
        random_state:int, 
        verbose:bool,
        fname:str):
    
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

        # if t == 0:
        #     print(f"Number of actions : {x.shape[0]}\tReward range : [{np.amin(exp_rewards):.5f}, {np.amax(exp_rewards):.5f}]")
        
        ## compute the optimal action
        optimal_action = np.argmax(exp_rewards)
        optimal_reward = exp_rewards[optimal_action]

        ## choose the best action
        noise = subgaussian_noise(distribution=noise_dist, 
                                  size=1, 
                                  std=noise_std, 
                                  random_state=random_state_)
        
        if isinstance(agent, ContextualBandit):
            chosen_action = agent.choose(x)
        else:
            chosen_action = agent.choose()
        chosen_reward = exp_rewards[chosen_action] + noise
        
        if verbose:
            try:
                string = f"""
                        case : {cfg.case}, SEED : {cfg.seed}, K : {cfg.arms}, 
                        Latent_dim : {cfg.latent_dim}, Obs_dim : {cfg.dim}, 
                        Trial : {trial}, p : {cfg.p}, Agent : {agent.__class__.__name__}, 
                        Round : {t+1}, optimal : {optimal_action}, a_hat: {agent.a_hat}, 
                        pseudo : {agent.pseudo_action}, chosen : {agent.chosen_action}
                    """
            except:
                string = f"""
                        case : {cfg.case}, SEED : {cfg.seed}, K : {cfg.arms}, 
                        Latent_dim : {cfg.latent_dim}, Obs_dim : {cfg.dim}, 
                        Trial : {trial}, p : {cfg.p}, Agent : {agent.__class__.__name__}, 
                        Round : {t+1}, optimal : {optimal_action}, chosen : {chosen_action}
                    """
            save_log(path=LOG_PATH, fname=fname, string=" ".join(string.split()))
            print(" ".join(string.split()))

        ## compute the regret
        regrets[t] = optimal_reward - exp_rewards[chosen_action]

        ## update the agent
        if isinstance(agent, ContextualBandit):
            agent.update(x=x, r=chosen_reward)
        else:
            agent.update(a=chosen_action, r=chosen_reward)

    return np.cumsum(regrets)


def show_result(regrets:dict, 
                horizon:int, 
                figsize:tuple=(6, 5), 
                fontsize=11):
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'black']
    period = horizon // 10
    
    z_init = len(colors)
    # Plot the graph for each algorithm with error bars
    for i, (color, (key, item)) in enumerate(zip(colors, regrets.items())):
        rounds = np.arange(horizon)
        mean = np.mean(item, axis=0)
        std = np.std(item, axis=0, ddof=1)
        
        # Display the line with markers and error bars periodically
        ax.errorbar(rounds[::period], mean[::period], yerr=std[::period], label=f"{key}", 
                    fmt='s', color=color, capsize=3, elinewidth=1, zorder=z_init-i)
        
        # Display the full line without periodic markers
        ax.plot(rounds, mean, color=color, linewidth=2, zorder=z_init-i)
    
    ax.grid(True)
    ax.set_xlabel(r"Round ($t$)")
    ax.set_ylabel("Cumulative Regret")
    ax.legend(loc="upper left", fontsize=fontsize)
    
    fig.tight_layout()  
    return fig


if __name__ == "__main__":
    ## hyper-parameters
    arms = cfg.arms # List[int]
    k = cfg.latent_dim
    d = cfg.dim
    T = cfg.horizon
    SEED = cfg.seed
    sigma = cfg.reward_std
    AGENTS = [
        "rolf_lasso", 
        "rolf_ridge", 
        "dr_lasso", 
        "linucb", 
        "lints", 
        "mab_ucb"
              ]
    case = cfg.case
    fname = f"Case_{case}_K_{arms}_k_{k}_d_{d}_T_{T}_explored_{cfg.init_explore}_noise_{sigma}"
   
    # regret_results = dict()
    # for agent_type in AGENTS:
    #     regrets = run_trials(agent_type=agent_type, 
    #                          trials=cfg.trials, 
    #                          horizon=T, 
    #                          k=k, 
    #                          d=d, 
    #                          arms=arms, 
    #                          noise_std=cfg.reward_std, 
    #                          random_state=SEED, 
    #                          verbose=True)
    #     key = AGENT_DICT[agent_type]
    #     regret_results[key] = regrets

    # Function to run trials for a single agent
    def run_agent(agent_type):
        regrets = run_trials(
            agent_type=agent_type, 
            trials=cfg.trials, 
            horizon=T, 
            k=k, 
            d=d, 
            arms=arms, 
            noise_std=cfg.reward_std,
            case=case,
            random_state=SEED, 
            verbose=True,
            fname=fname
        )
        key = AGENT_DICT[agent_type]
        return key, regrets

    # Parallel execution using ProcessPoolExecutor
    regret_results = dict()
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = executor.map(run_agent, AGENTS)

    # Collect results
    for key, regrets in results:
        regret_results[key] = regrets
    
    fig = show_result(regrets=regret_results, 
                      horizon=T, 
                      fontsize=15)

    save_plot(fig, 
              path=FIGURE_PATH, 
              fname=fname)
    save_result(result=(vars(cfg), regret_results), 
                path=RESULT_PATH, 
                fname=fname, 
                filetype=cfg.filetype)