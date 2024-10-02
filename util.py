import os
import pickle
import json
import numpy as np
from typing import Union, List, Tuple, Dict
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from datetime import datetime

def generate_uniform(dim:Union[int, tuple], uniform_rng:list=None):
    ## Generates a given dimension of random vector that where each element follows the uniform distribution in a given range
    assert type(dim) == int or type(dim) == tuple, "The type of 'dim' must be either int or tuple."

    if uniform_rng is None:
        low, high = -1., 1.
    else:
        assert len(uniform_rng) == 2, "The 'uniform_rng' must contain two elements: low and high."
        low, high = uniform_rng

    if type(dim) == int:
        size = dim
    else:
        dim1, dim2 = dim
        size = dim1 * dim2
    return np.random.uniform(low=low, high=high, size=size).reshape(dim)


def shermanMorrison(V:np.ndarray, x:np.ndarray):
    """
    V: inverse of old gram matrix, corresponding to $V_{t-1}$.
    x: a new observed context
    return: inverse of new gram matrix
    """
    numerator = np.einsum("ij, j, k, kl -> il", V, x, x, V)
    denominator = (1 + np.einsum("i, ij, j ->", x, V, x))
    return V - (numerator / denominator)


def vector_norm(v:np.ndarray, type:str):
    assert type in ["l1", "l2", "lsup"], "Type of the vector norm should be one of 'l1', 'l2', and 'lsup'."
    type_dict = {"l1": 1, "l2": 2, "lsup": np.inf}
    v = v.flatten()
    return np.linalg.norm(v, ord=type_dict[type])


def covariance_generator(d:int, independent:bool, distribution:str=None, uniform_rng:list=None, variances:Union[list, np.ndarray]=None):
    ## Generates a random covariance matrix
    if independent:
        if variances is None:
            assert distribution is not None and distribution.lower() in ["gaussian", "uniform"], "If the variances are not given, you need to pass the distribution to sample them."
            ## then variances are sampled randomly
            if distribution == "gaussian":
                variances = (np.random.randn(d)) ** 2
            else:
                variances = (generate_uniform(dim=d, uniform_rng=uniform_rng)) ** 2

        mat = np.zeros(shape=(d, d))
        for i in range(d):
            mat[i, i] = variances[i]

    else:
        assert distribution is not None and distribution.lower() in ["gaussian", "uniform"], f"If independent is {independent}, you need to pass the distribution to sample them."
        if distribution == "gaussian":
            rnd = np.random.randn(d, d)
        elif distribution == "uniform":
            rnd = generate_uniform(dim=(d, d), uniform_rng=uniform_rng)

        ## make a symmetric matrix
        sym = (rnd + rnd.T) / 2
        ## make positive semi-definite and bound its maximum singular value
        mat = sym @ sym.T
        if variances is not None:
            for i in range(d):
                mat[i, i] = variances[i]
    return mat


def gram_schmidt(A):
    ## Gram-Schmidt decomposition
    Q = np.zeros(A.shape)
    for i in range(A.shape[1]):
        # Orthogonalize the vector
        Q[:,i] = A[:,i]
        for j in range(i):
            Q[:,i] -= np.dot(Q[:,j], A[:,i]) * Q[:,j]

        # Normalize the vector
        Q[:,i] = Q[:,i] / np.linalg.norm(Q[:,i])
    return Q


def make_diagonal(v:np.ndarray, dim:Union[int, tuple]):
    ## Generate a diagonal matrix
    if type(dim) == int:
        diag = np.zeros((dim, dim))
        rng = dim
    else:
        diag = np.zeros(dim)
        rng = min(dim)

    for i in range(rng):
        diag[i, i] = v[i]
    return diag


def positive_definite_generator(dimension:int, distribution:str="uniform", uniform_rng:list=None):
    d = dimension

    ## create orthogonal eigenvectors via Gram-Schmidt process
    if distribution == "uniform":
        source = generate_uniform(dim=(d, d), uniform_rng=uniform_rng)
    else:
        source = np.random.randn(d, d)
    eigvecs = gram_schmidt(source)

    ## create a matrix of eigenvalues
    eigvals = generate_uniform(dim=d, uniform_rng=(0, 1))
    eigmat = make_diagonal(np.absolute(eigvals))

    ## make the targeted positive definite matrix
    Z = eigvecs @ eigmat @ eigvecs.T
    return Z


def minmax(v:np.ndarray, bound:float=1.):
    # minmax scaler
    min = np.min(v)
    max = np.max(v)
    return ((v - min) / (max - min)) * bound


def left_pseudo_inverse(A:np.ndarray):
    ## Perform SVD to obtain the left pseudo inverse
    d, k = A.shape
    u, A_sig, v_T = np.linalg.svd(A)

    B_sig = np.zeros((k, d))
    for i in range(min(d, k)):
        B_sig[i, i] = 1 / A_sig[i]
    B = v_T.T @ B_sig @ u.T
    return B


def rademacher(size:int):
    ## generate a Rademacher random variable
    return 2 * np.random.randint(0, 2, size) - 1


def subgaussian_noise(distribution:str, size:int, std:float=None, random_state:int=None):
    ## SubGaussian noise generator
    if random_state:
        np.random.seed(random_state)

    if distribution == "gaussian":
        if std is None:
            std = 1.
        noise = np.random.normal(loc=0, scale=std, size=size)
    elif distribution == "uniform":
        if std is None:
            uniform_rng = [-1., 1.]
        else:
            low = -np.sqrt(3) * std
            high = np.sqrt(3) * std
            uniform_rng = [low, high]
        noise = generate_uniform(dim=size, uniform_rng=uniform_rng)
    else:
        std = 1.
        noise = rademacher(size=size)

    if size == 1:
        return noise[0]
    elif size == 0:
        return 0
    return noise


def bounding(type:str, v:np.ndarray, bound:float, method:str=None, norm_type:str=None):
    ## Function to bound a vector or a matrix
    type_dict = {"l1": 1, "l2": 2, "lsup": np.inf}
    if type == "param":
        assert norm_type is not None, "For a vector, you should input which type of norm you are going to use."
        if vector_norm(v, type=norm_type) > bound:
            v *= (bound / vector_norm(v, norm_type))
    elif type == "feature":
        assert method in ["scaling", "clipping"], f"If you're trying to bound {type}, the method should not be None."
        if method == "scaling":
            maxnorm = np.max([vector_norm(item, type=norm_type) for item in v])
            v *= (bound / maxnorm)
        else:
            norms = np.linalg.norm(v, axis=1, ord=type_dict[norm_type])
            scale_factors = np.where(norms > bound, bound / norms, 1)
            v = v * scale_factors[:, np.newaxis]  # Scale each row without loop
    elif type == "mapping":
        assert method in ["lower", "upper"], f"If you're trying to bound {type}, you need to specify the lower or the upper bound."
        if method == "lower":
            ## constrain the lower bound of the minimum singular value
            u, sig, v_T = np.linalg.svd(v)
            sig = sig - np.min(sig) + bound
            sig_v = make_diagonal(sig, dim=v.shape)
            v = u @ sig_v @ v_T

        if method == "upper":
            ## constrain the upper bound of the spectral norm
            v *= (bound / np.linalg.norm(v, 2))
    return v


def feature_sampler(dimension:int, feat_dist:str, size:int, disjoint:bool, cov_dist:str=None, bound:float=None,
                    bound_method:str=None, bound_type:str=None, uniform_rng:list=None, random_state:int=None):
    ## Function to sample a feature matrix
    assert feat_dist.lower() in ["gaussian", "uniform"], "Feature distribution must be either 'gaussian' or 'uniform'."
    if random_state:
        np.random.seed(random_state)

    if disjoint:
        if feat_dist.lower() == "gaussian":
            assert uniform_rng is None, f"If the distribution is {feat_dist}, variable range is not required."
            ## gaussian
            variances = np.ones(dimension)
            cov = covariance_generator(d=dimension, independent=True, variances=variances)
            feat = np.random.multivariate_normal(mean=np.zeros(dimension), cov=cov, size=size)
        else:
            ## uniform
            feat = generate_uniform(dim=(size, dimension), uniform_rng=uniform_rng)
    else:
        assert cov_dist is not None, f"If 'disjoint' is set to {disjoint}, it is required to specify the distribution to sample the covariance matrix."
        if feat_dist.lower() == "gaussian":
            ## gaussian
            cov = covariance_generator(d=dimension, independent=False, distribution=cov_dist)
            feat = np.random.multivariate_normal(mean=np.zeros(dimension), cov=cov, size=size)
        else:
            ## uniform
            feat = generate_uniform(dim=(size, dimension), uniform_rng=uniform_rng)

            # Cholesky decomposition
            pd = positive_definite_generator(dimension=dimension, distribution=cov_dist)
            L = np.linalg.cholesky(pd)
            for i in range(size):
                feat[i, :] = L @ feat[i, :]

    # Ensure the matrix is full-rank by adding random noise if necessary
    while np.linalg.matrix_rank(feat) < min(size, dimension):
        feat += np.random.normal(0, 1e-4, size=feat.shape)

    if bound is not None:
        assert bound_method in ["scaling", "clipping"], "Bounding method should either be 'scaling' or 'clipping'."
        assert bound_type in ["l1", "l2", "lsup"], "Bounding type must be one of 'l1', 'l2', 'lsup'."
        feat = bounding(type="feature", v=feat, bound=bound, method=bound_method, norm_type=bound_type)
    return feat


def mapping_generator(latent_dim:int, obs_dim:int, distribution:str, lower_bound:float=None, upper_bound:float=None, uniform_rng:list=None, random_state:int=None):
    ## Function that generates a linear mapping
    assert distribution.lower() in ["gaussian", "uniform"], "Feature distribution must be either 'gaussian' or 'uniform'."
    if random_state:
        np.random.seed(random_state)

    if distribution.lower() == "gaussian":
        assert uniform_rng is None, f"If the distribution is {distribution}, variable range is not required."
        mat = np.random.randn(obs_dim, latent_dim)
    else:
        if uniform_rng is None:
            mat = generate_uniform(dim=(obs_dim, latent_dim), uniform_rng=[-np.sqrt(2/latent_dim), np.sqrt(2/latent_dim)])
        else:
            mat = generate_uniform(dim=(obs_dim, latent_dim), uniform_rng=uniform_rng)

    if lower_bound is not None:
        ## constrain the lower bound of the spectral norm
        mat = bounding(type="mapping", v=mat, bound=lower_bound, method="lower")

    if upper_bound is not None:
        ## constrain the upper bound of the spectral norm
        mat = bounding(type="mapping", v=mat, bound=upper_bound, method="upper")
    return mat


def param_generator(dimension:int, distribution:str, disjoint:bool, bound:float=None, bound_type:str=None, uniform_rng:list=None, random_state:int=None):
    ## Function that generates an unknown parameter
    assert distribution.lower() in ["gaussian", "uniform"], "Parameter distribution must be either 'gaussian' or 'uniform'."
    if random_state:
        np.random.seed(random_state)

    if disjoint:
        if distribution == "gaussian":
            assert uniform_rng is None, f"If the distribution is {distribution}, variable range is not required."
            param = np.random.randn(dimension)
        else:
            param = generate_uniform(dim=dimension, uniform_rng=uniform_rng)
    else:
        if distribution == "gaussian":
            cov = covariance_generator(dimension, distribution=distribution)
            param = np.random.multivariate_normal(mean=np.zeros(dimension), cov=cov)
        else:
            # uniform
            param = generate_uniform(dim=dimension, uniform_rng=uniform_rng)
            pd = positive_definite_generator(dimension, distribution=distribution)
            L = np.linalg.cholesky(pd)
            param = L @ param

    if bound is not None:
        assert bound_type in ["l1", "l2", "lsup"], "Bounding type must be one of 'l1', 'l2', 'lsup'."
        param = bounding(type="param", v=param, bound=bound, norm_type=bound_type)
    return param


def save_plot(fig:Figure, path:str, fname:str, extension:str="pdf"):
    os.makedirs(path, exist_ok=True)
    fig.savefig(f"{path}/{fname}.{extension}")
    print("Plot is Saved Completely!")


def save_result(result:dict, path:str, fname:str, filetype:str):
    assert filetype in ["pickle", "json"]
    os.makedirs(path, exist_ok=True)

    if filetype == "pickle":
        with open(f"{path}/{fname}.pkl", "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif filetype == "json":
        with open(f"{path}/{fname}.json", "w") as f:
            json.dump(result, f)

    print("Result is Saved Completely!")


def orthogonal_complement_basis(X):
    d, K = X.shape
    
    # Perform Singular Value Decomposition
    _, _, Vt = np.linalg.svd(X)

    # Find the rank of X to determine the number of non-zero singular values
    rank = np.linalg.matrix_rank(X)

    # The basis for the null space (orthogonal complement of the row space)
    # is given by the columns of V corresponding to zero singular values
    if d <= K:
        null_space_basis = Vt[rank:].T
    else:
        null_space_basis = Vt.T

    return null_space_basis