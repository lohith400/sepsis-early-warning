import numpy as np

from sepsis.anfis import class_weights, decode, lse, phi, score


def test_rule_strengths_are_normalised():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    C, S = decode(np.concatenate([rng.normal(size=8 * 3), np.zeros(8 * 3)]), 8, 3)
    w, P = phi(X, C, S)
    np.testing.assert_allclose(w.sum(1), 1.0, atol=1e-9)
    assert P.shape == (50, 8 * 4)


def test_least_squares_recovers_a_linear_function():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 2)); y = 0.5 + 2 * X[:, 0] - X[:, 1]
    C, S = np.zeros((1, 2)), np.full((1, 2), 3.0)           # one rule covers everything
    _, P = phi(X, C, S)
    th = lse(P, y, np.ones(len(y)), lam=1e-6)
    np.testing.assert_allclose(score(X, C, S, th), y, atol=1e-3)


def test_class_weights_balance_the_classes():
    y = np.array([0] * 90 + [1] * 10, float)
    w = class_weights(y)
    assert abs(w[y == 1].sum() - w[y == 0].sum()) < 1e-9
