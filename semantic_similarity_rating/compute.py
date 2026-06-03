"""
Utility functions for computing and manipulating probability density functions (PMFs) and embeddings.

This module provides functions for:
- Converting between different similarity metrics (cosine, KS)
- Scaling PMFs using temperature parameters
- Computing statistical moments of PMFs
- Finding optimal temperature parameters for PMF scaling
- Converting response embeddings to PMFs

The module is particularly useful for working with Likert scale responses and their
embeddings, providing tools to analyze and transform the underlying probability
distributions.
"""

import numpy as np


def scale_pmf(pmf, temperature, max_temp=np.inf):
    """
    Scale a PMF using temperature scaling.

    Parameters
    ----------
    pmf : array_like
        Input probability density function
    temperature : float
        Temperature parameter for scaling (0 to max_temp)
    max_temp : float, optional
        Maximum temperature value, by default np.inf

    Returns
    -------
    numpy.ndarray
        Scaled PMF where all values sum to 1

    Notes
    -----
    - If temperature is 0, returns a one-hot vector at the maximum probability
    - If temperature > max_temp, uses max_temp for scaling
    - Otherwise uses the specified temperature for scaling
    """
    if temperature == 0.0:
        if np.all(pmf == pmf[0]):
            return pmf
        else:
            new_pmf = np.zeros_like(pmf)
            new_pmf[np.argmax(pmf)] = 1.0
            return new_pmf
    elif temperature > max_temp:
        hist = pmf ** (1 / max_temp)
    else:
        hist = pmf ** (1 / temperature)
    return hist / hist.sum()


def scale_pmfs(pmfs, temperature, max_temp=np.inf):
    """
    Scale a batch of PMFs using temperature scaling (vectorized).

    Parameters
    ----------
    pmfs : array_like, shape (n, k)
        Stack of probability mass functions, one per row.
    temperature : float
        Temperature parameter for scaling (0 to max_temp).
    max_temp : float, optional
        Maximum temperature value, by default np.inf

    Returns
    -------
    numpy.ndarray, shape (n, k)
        Scaled PMFs where each row sums to 1.

    Notes
    -----
    Semantically equivalent to applying ``scale_pmf`` to each row, but
    implemented with numpy broadcasting to avoid per-row Python overhead.
    """
    pmfs = np.asarray(pmfs)
    if pmfs.ndim == 1:
        return scale_pmf(pmfs, temperature, max_temp=max_temp)

    if temperature == 0.0:
        # One-hot at argmax per row; pass-through rows that are uniform.
        out = np.zeros_like(pmfs)
        out[np.arange(pmfs.shape[0]), np.argmax(pmfs, axis=1)] = 1.0
        uniform_rows = np.all(pmfs == pmfs[:, :1], axis=1)
        out[uniform_rows] = pmfs[uniform_rows]
        return out

    t = min(temperature, max_temp)
    hist = pmfs ** (1 / t)
    return hist / hist.sum(axis=1, keepdims=True)


def cosine_similarity_matrix(matrix_responses, matrix_likert_sentences):
    """
    Compute the (1 + cosine) / 2 similarity between response and Likert embeddings.

    Parameters
    ----------
    matrix_responses : array_like, shape (n_responses, dim)
        Response embeddings, one per row.
    matrix_likert_sentences : array_like, shape (dim, n_likert_points)
        Likert anchor embeddings, one per column.

    Returns
    -------
    numpy.ndarray, shape (n_responses, n_likert_points)
        Similarity matrix in [0, 1]: γ(r, ℓ) = (1 + cosine(r, ℓ)) / 2.

    Notes
    -----
    Kept as a separate step so callers sweeping over (temperature, epsilon)
    grids can precompute similarities once and pass them to
    ``similarities_to_pmf`` / ``scale_pmfs`` in the inner loop.
    """
    M_left = matrix_responses
    M_right = matrix_likert_sentences

    if M_left.shape[0] == 0:
        return np.empty((0, M_right.shape[1]))

    norm_right = np.linalg.norm(M_right, axis=0)
    M_right = M_right / norm_right[None, :]

    norm_left = np.linalg.norm(M_left, axis=1)
    M_left = M_left / norm_left[:, None]

    return (1 + M_left.dot(M_right)) / 2


def precompute_similarity_stats(cos):
    """
    Precompute per-row reductions of a cosine similarity matrix.

    Amortizes the row-wise min, sum, and argmin over an outer sweep of
    ``(temperature, epsilon)`` pairs — each is otherwise recomputed on
    every call inside ``similarities_to_pmf``.

    Parameters
    ----------
    cos : array_like, shape (n_responses, n_likert_points)
        Output of ``cosine_similarity_matrix``.

    Returns
    -------
    dict
        Keys: ``cos`` (the input), ``cos_min`` (shape (n, 1)),
        ``cos_sum`` (shape (n, 1)), ``min_indices`` (shape (n,)).
    """
    cos = np.asarray(cos)
    return {
        "cos": cos,
        "cos_min": cos.min(axis=1, keepdims=True),
        "cos_sum": cos.sum(axis=1, keepdims=True),
        "min_indices": np.argmin(cos, axis=1),
    }


def similarities_to_pmf(similarities, epsilon=0.0):
    """
    Apply the SSR numerator/denominator formula to precomputed similarities.

    Parameters
    ----------
    similarities : array_like or dict
        Either a raw similarity matrix (shape ``(n_responses, n_likert_points)``)
        or a dict produced by ``precompute_similarity_stats``. The dict form
        avoids recomputing the row-wise min/sum/argmin for every call — useful
        when sweeping over an ``epsilon`` grid.
    epsilon : float, optional
        Regularization added at the minimum-similarity position and to the
        denominator. Default 0.0.

    Returns
    -------
    numpy.ndarray, shape (n_responses, n_likert_points)
        Per-response PMF over the Likert scale.
    """
    if isinstance(similarities, dict):
        cos = similarities["cos"]
        cos_min = similarities["cos_min"]
        cos_sum = similarities["cos_sum"]
        min_indices = similarities["min_indices"]
    else:
        cos = np.asarray(similarities)
        if cos.shape[0] == 0:
            return np.empty_like(cos)
        cos_min = cos.min(axis=1, keepdims=True)
        cos_sum = cos.sum(axis=1, keepdims=True)
        min_indices = None  # Computed lazily if epsilon > 0.

    if cos.shape[0] == 0:
        return np.empty_like(cos)

    numerator = cos - cos_min
    if epsilon > 0:
        if min_indices is None:
            min_indices = np.argmin(cos, axis=1)
        numerator[np.arange(cos.shape[0]), min_indices] += epsilon

    n_likert_points = cos.shape[1]
    denominator = cos_sum - n_likert_points * cos_min + epsilon
    return numerator / denominator


def response_embeddings_to_pmf(matrix_responses, matrix_likert_sentences, epsilon=0.0):
    """
    Convert response embeddings and Likert sentence embeddings to a PMF.

    Parameters
    ----------
    matrix_responses : array_like
        Matrix of response embeddings
    matrix_likert_sentences : array_like
        Matrix of Likert sentence embeddings
    epsilon : float, optional
        Small regularization parameter to prevent division by zero and add smoothing.
        Default is 0.0 (no regularization).

    Returns
    -------
    numpy.ndarray
        Probability density function representing the response distribution

    Notes
    -----
    This implements the SSR equation:
    p_{c,i}(r) = [γ(σ_{r,i}, t_c̃) - γ(σ_ℓ,i, t_c̃) + ε δ_ℓ,r] /
                 [Σ_r γ(σ_{r,i}, t_c̃) - n_points * γ(σ_ℓ,i, t_c̃) + ε]
    where γ is the cosine similarity function, δ_ℓ,r is the Kronecker delta,
    and n_points is the number of Likert scale points.

    When sweeping (temperature, epsilon), prefer precomputing the cosine
    matrix with ``cosine_similarity_matrix`` and reusing it via
    ``similarities_to_pmf`` — this avoids redundant normalization and matmul.
    """
    cos = cosine_similarity_matrix(matrix_responses, matrix_likert_sentences)
    return similarities_to_pmf(cos, epsilon=epsilon)
