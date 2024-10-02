import numpy as np

def oful_alpha(maxnorm:float, horizon:int, d:int, arms:int, lbda:float, reward_std:float, context_std:float) -> float:
    # first term
    log_num = (maxnorm ** 2) * horizon
    # log_denom = (d + arms) * lbda
    log_denom = d * lbda
    # alpha_1 = reward_std * np.sqrt((d + arms) * np.log(1 + (log_num / log_denom)))
    alpha_1 = reward_std * np.sqrt(d * np.log(1 + (log_num / log_denom)))
    
    # second term
    alpha_2 = np.sqrt(lbda)
    
    # third term
    numerator = horizon * d * d * np.log(horizon * d)
    denominator = lbda
    alpha_3 = context_std * np.sqrt(numerator / denominator)
    
    alpha = (alpha_1 + alpha_2 + alpha_3)
    return alpha 

def linucb_alpha(delta:float) -> float:
    return 1 + np.sqrt(np.log(2/delta)/2)

def lints_alpha(d:int, reward_std:float, delta:float, epsilon:float=0.5) -> float:
    return reward_std * np.sqrt((24 / epsilon) * d * np.log(1 / delta))

def hop_alpha(lbda:float, horizon:int, reward_std:float, arms:int, delta:float):
    root = np.sqrt(2 * np.log(2 * horizon * arms / delta))
    return (reward_std * root) + (2 * np.sqrt(lbda))

# def ucbglm_alpha()

if __name__ == "__main__":
    T = 3000
    maxnorm = 1
    d, arms = 10, 20
    lbda = 1
    delta = 0.00001
    reward_std = 0.1
    # context_std = [(t+1) ** (-0.5) for t in range(T)]
    context_std = T ** (-0.5)
    
    print("alpha comparison")
    for t in range(T):
        print(f"Time : {t}", end="\t")
        print(f"linucb : {linucb_alpha(delta) * np.sqrt(np.log(t)):.5f}", end='\t')
        print(f"lints : {lints_alpha(d=d, reward_std=reward_std, delta=delta):.5f}", end="\t")
        print(f"oful : {oful_alpha(maxnorm, T, d, arms, lbda, reward_std, context_std):.5f}")
        print(f"hop : {hop_alpha(lbda=lbda, horizon=T, reward_std=reward_std, arms=arms, delta=delta):.5f}")
