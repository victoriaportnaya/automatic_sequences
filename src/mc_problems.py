#!/usr/bin/env python3
"""Monte Carlo / QMC benchmark problems with exact references."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from finance_mc_low_noise import bs_call_price, call_payoff_from_uniforms
from finance_qmc_nc_experiment import gbm_arithmetic_asian_call_from_uniforms, ndtri


@dataclass(frozen=True)
class MCProblem:
    name: str
    category: str
    description: str
    default_dim: int
    exact: Callable[..., float]
    evaluate: Callable[[np.ndarray], np.ndarray]
    aggregate: Callable[[np.ndarray], float] | None = None

    def estimate(self, u: np.ndarray) -> float:
        vals = self.evaluate(u)
        if self.aggregate is not None:
            return float(self.aggregate(vals))
        return float(np.mean(vals))


def _geometric_asian_exact(
    s0: float, k: float, r: float, sigma: float, t: float, n_steps: int
) -> float:
    dt = t / n_steps
    times = np.arange(1, n_steps + 1, dtype=np.float64) * dt
    mean_log_g = math.log(s0) + (r - 0.5 * sigma * sigma) * float(np.mean(times))
    mins = np.minimum.outer(times, times)
    var_log_g = sigma * sigma * float(np.sum(mins)) / float(n_steps * n_steps)
    std_log_g = math.sqrt(var_log_g)
    if std_log_g <= 0.0:
        return max(math.exp(mean_log_g) - k, 0.0) * math.exp(-r * t)
    d1 = (mean_log_g - math.log(k) + var_log_g) / std_log_g
    d2 = d1 - std_log_g
    phi_d1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    phi_d2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
    return math.exp(-r * t) * (math.exp(mean_log_g + 0.5 * var_log_g) * phi_d1 - k * phi_d2)


def _geometric_asian_payoff(u: np.ndarray, s0: float, k: float, r: float, sigma: float, t: float) -> np.ndarray:
    n_paths, n_steps = u.shape
    z = ndtri(np.clip(u, 1e-12, 1 - 1e-12))
    dt = t / n_steps
    drift = (r - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    log_s = np.full(n_paths, math.log(s0), dtype=np.float64)
    sum_log_s = np.zeros(n_paths, dtype=np.float64)
    for j in range(n_steps):
        log_s += drift + vol * z[:, j]
        sum_log_s += log_s
    g = np.exp(sum_log_s / float(n_steps))
    return np.exp(-r * t) * np.maximum(g - k, 0.0)


def _normal_tail_prob(threshold: float) -> float:
    return 0.5 * math.erfc(threshold / math.sqrt(2.0))


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _lazy_exact(fn: Callable[[], float]) -> Callable[[], float]:
    cache: list[float | None] = [None]

    def wrapped() -> float:
        if cache[0] is None:
            cache[0] = float(fn())
        return cache[0]

    return wrapped


def _bs_d1_d2(s0: float, k: float, r: float, sigma: float, t: float) -> tuple[float, float]:
    d1 = (math.log(s0 / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    return d1, d1 - sigma * math.sqrt(t)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_put_price(s0: float, k: float, r: float, sigma: float, t: float) -> float:
    if sigma <= 0.0 or t <= 0.0:
        return max(k * math.exp(-r * t) - s0, 0.0)
    d1, d2 = _bs_d1_d2(s0, k, r, sigma, t)
    return float(k * math.exp(-r * t) * _normal_cdf(-d2) - s0 * _normal_cdf(-d1))


def bs_digital_call_price(s0: float, k: float, r: float, sigma: float, t: float, payout: float = 1.0) -> float:
    if sigma <= 0.0 or t <= 0.0:
        return payout * math.exp(-r * t) * (1.0 if s0 > k else 0.0)
    _, d2 = _bs_d1_d2(s0, k, r, sigma, t)
    return float(payout * math.exp(-r * t) * _normal_cdf(d2))


def _gbm_terminal(u: np.ndarray, s0: float, r: float, sigma: float, t: float) -> np.ndarray:
    z = ndtri(np.clip(u.reshape(-1), 1e-12, 1 - 1e-12))
    return s0 * np.exp((r - 0.5 * sigma * sigma) * t + sigma * math.sqrt(t) * z)


def put_payoff_from_uniforms(u: np.ndarray, s0: float, k: float, r: float, sigma: float, t: float) -> np.ndarray:
    st = _gbm_terminal(u, s0, r, sigma, t)
    return np.exp(-r * t) * np.maximum(k - st, 0.0)


def digital_call_payoff_from_uniforms(
    u: np.ndarray, s0: float, k: float, r: float, sigma: float, t: float, payout: float = 1.0
) -> np.ndarray:
    st = _gbm_terminal(u, s0, r, sigma, t)
    return payout * math.exp(-r * t) * (st > k).astype(np.float64)


def _one_dim_gaussian_cube() -> float:
    return 0.5 * math.sqrt(math.pi) * math.erf(1.0)


def make_finance_problems(
    s0: float = 100.0,
    strike: float = 100.0,
    rate: float = 0.03,
    sigma: float = 0.2,
    maturity: float = 1.0,
    n_steps: int = 32,
    ref_asian: float | None = None,
) -> dict[str, MCProblem]:
    kw = dict(s0=s0, k=strike, r=rate, sigma=sigma, t=maturity)

    def european_exact() -> float:
        return bs_call_price(s0, strike, rate, sigma, maturity)

    def european_eval(u: np.ndarray) -> np.ndarray:
        return call_payoff_from_uniforms(u.reshape(-1), **kw)

    def arith_asian_exact() -> float:
        if ref_asian is not None:
            return ref_asian
        rng = np.random.default_rng(4242)
        u = rng.random((1 << 18, n_steps))
        return float(np.mean(gbm_arithmetic_asian_call_from_uniforms(u, **kw)))

    arith_asian_exact_lazy = _lazy_exact(arith_asian_exact)

    def arith_asian_eval(u: np.ndarray) -> np.ndarray:
        return gbm_arithmetic_asian_call_from_uniforms(u, **kw)

    def geom_asian_exact() -> float:
        return _geometric_asian_exact(s0, strike, rate, sigma, maturity, n_steps)

    def geom_asian_eval(u: np.ndarray) -> np.ndarray:
        return _geometric_asian_payoff(u, **kw)

    return {
        "european_call": MCProblem(
            name="european_call",
            category="finance",
            description="European call under GBM; exact Black–Scholes",
            default_dim=1,
            exact=european_exact,
            evaluate=european_eval,
        ),
        "european_put": MCProblem(
            name="european_put",
            category="finance",
            description="European put under GBM; exact Black–Scholes",
            default_dim=1,
            exact=lambda: bs_put_price(s0, strike, rate, sigma, maturity),
            evaluate=lambda u: put_payoff_from_uniforms(u.reshape(-1), **kw),
        ),
        "digital_call": MCProblem(
            name="digital_call",
            category="finance",
            description="Cash-or-nothing call; exact digital BS",
            default_dim=1,
            exact=lambda: bs_digital_call_price(s0, strike, rate, sigma, maturity),
            evaluate=lambda u: digital_call_payoff_from_uniforms(u.reshape(-1), **kw),
        ),
        "arithmetic_asian_call": MCProblem(
            name="arithmetic_asian_call",
            category="finance",
            description=f"Arithmetic-mean Asian call ({n_steps} steps); high-N IID reference",
            default_dim=n_steps,
            exact=arith_asian_exact_lazy,
            evaluate=arith_asian_eval,
        ),
        "geometric_asian_call": MCProblem(
            name="geometric_asian_call",
            category="finance",
            description=f"Geometric-mean Asian call ({n_steps} steps); closed-form",
            default_dim=n_steps,
            exact=geom_asian_exact,
            evaluate=geom_asian_eval,
        ),
    }


def make_integral_problems(dimension: int = 12) -> dict[str, MCProblem]:
    d = dimension

    def exp_sum_exact() -> float:
        return float((math.e - 1.0) ** d)

    def exp_sum_eval(x: np.ndarray) -> np.ndarray:
        return np.exp(np.sum(x, axis=1))

    def indicator_exact() -> float:
        return 0.5

    def indicator_eval(x: np.ndarray) -> np.ndarray:
        return (np.mean(x, axis=1) > 0.5).astype(np.float64)

    def sin_product_exact() -> float:
        return float((2.0 / math.pi) ** d)

    def sin_product_eval(x: np.ndarray) -> np.ndarray:
        return np.prod(np.sin(math.pi * x), axis=1)

    def poly_corner_exact() -> float:
        # ∫_{[0,1]^d} (x_1 + ... + x_d) dx = d/2
        return float(d) / 2.0

    def poly_corner_eval(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    def gaussian_bump_exact() -> float:
        rng = np.random.default_rng(777)
        xs = rng.random((1 << 20, d))
        return float(np.mean(np.exp(-np.sum((xs - 0.5) ** 2, axis=1))))

    gaussian_bump_exact_lazy = _lazy_exact(gaussian_bump_exact)

    def gaussian_bump_eval(x: np.ndarray) -> np.ndarray:
        return np.exp(-np.sum((x - 0.5) ** 2, axis=1))

    def gaussian_sphere_exact() -> float:
        return float(_one_dim_gaussian_cube() ** d)

    def gaussian_sphere_eval(x: np.ndarray) -> np.ndarray:
        return np.exp(-np.sum(x * x, axis=1))

    def g_function_exact() -> float:
        return 1.0

    def g_function_eval(x: np.ndarray) -> np.ndarray:
        # Sobol g-function with a_i=1: each 1D factor integrates to 1.
        return np.prod(2.0 * np.abs(2.0 * x - 1.0), axis=1)

    def max_coordinate_exact() -> float:
        # P(max x_i > 0.9) = 1 - (0.9)^d on [0,1]^d with independent uniform.
        return float(1.0 - 0.9**d)

    def max_coordinate_eval(x: np.ndarray) -> np.ndarray:
        return (np.max(x, axis=1) > 0.9).astype(np.float64)

    def l2_ball_indicator_exact() -> float:
        # Volume of {x in [0,1]^d : ||x||_2 <= 0.85} — high-N reference.
        rng = np.random.default_rng(888)
        xs = rng.random((1 << 20, d))
        return float(np.mean(np.linalg.norm(xs, axis=1) <= 0.85))

    l2_ball_exact_lazy = _lazy_exact(l2_ball_indicator_exact)

    def l2_ball_indicator_eval(x: np.ndarray) -> np.ndarray:
        return (np.linalg.norm(x, axis=1) <= 0.85).astype(np.float64)

    return {
        "exp_sum": MCProblem(
            name="exp_sum",
            category="integral",
            description=f"Smooth: exp(Σx_i) on [0,1]^{d}; exact (e-1)^{d}",
            default_dim=d,
            exact=exp_sum_exact,
            evaluate=exp_sum_eval,
        ),
        "indicator_mean": MCProblem(
            name="indicator_mean",
            category="integral",
            description=f"Discontinuous: 1{{mean(x)>0.5}} on [0,1]^{d}; exact 1/2",
            default_dim=d,
            exact=indicator_exact,
            evaluate=indicator_eval,
        ),
        "sin_product": MCProblem(
            name="sin_product",
            category="integral",
            description=f"Smooth periodic: ∏sin(πx_i) on [0,1]^{d}; exact (2/π)^{d}",
            default_dim=d,
            exact=sin_product_exact,
            evaluate=sin_product_eval,
        ),
        "poly_sum": MCProblem(
            name="poly_sum",
            category="integral",
            description=f"Linear: Σx_i on [0,1]^{d}; exact d/2",
            default_dim=d,
            exact=poly_corner_exact,
            evaluate=poly_corner_eval,
        ),
        "gaussian_bump": MCProblem(
            name="gaussian_bump",
            category="integral",
            description=f"Localized: exp(-||x-0.5||²) on [0,1]^{d}; high-N reference",
            default_dim=d,
            exact=gaussian_bump_exact_lazy,
            evaluate=gaussian_bump_eval,
        ),
        "gaussian_sphere": MCProblem(
            name="gaussian_sphere",
            category="integral",
            description=f"Smooth: exp(-||x||²) on [0,1]^{d}; product of 1D integrals",
            default_dim=d,
            exact=gaussian_sphere_exact,
            evaluate=gaussian_sphere_eval,
        ),
        "g_function": MCProblem(
            name="g_function",
            category="integral",
            description=f"Sobol g-function (a_i=1) on [0,1]^{d}; exact 1",
            default_dim=d,
            exact=g_function_exact,
            evaluate=g_function_eval,
        ),
        "max_coordinate": MCProblem(
            name="max_coordinate",
            category="integral",
            description=f"Discontinuous: 1{{max x_i > 0.9}} on [0,1]^{d}; exact 1-0.9^d",
            default_dim=d,
            exact=max_coordinate_exact,
            evaluate=max_coordinate_eval,
        ),
        "l2_ball_indicator": MCProblem(
            name="l2_ball_indicator",
            category="integral",
            description=f"Discontinuous: 1{{||x||_2 <= 0.85}} on [0,1]^{d}; high-N reference",
            default_dim=d,
            exact=l2_ball_exact_lazy,
            evaluate=l2_ball_indicator_eval,
        ),
    }


def make_risk_problems(
    rare_dim: int = 16,
    rare_threshold: float = 2.5,
    es_alpha: float = 0.99,
) -> dict[str, MCProblem]:
    q_es = float(ndtri(np.array([es_alpha], dtype=np.float64))[0])

    def rare_exact() -> float:
        return _normal_tail_prob(rare_threshold)

    def rare_eval(u: np.ndarray) -> np.ndarray:
        z = ndtri(np.clip(u, 1e-12, 1 - 1e-12))
        s = np.mean(z, axis=1) * math.sqrt(u.shape[1])
        return (s > rare_threshold).astype(np.float64)

    def es_exact() -> float:
        return _normal_pdf(q_es) / (1.0 - es_alpha)

    def es_eval(u: np.ndarray) -> np.ndarray:
        z = ndtri(np.clip(u.reshape(-1), 1e-12, 1 - 1e-12))
        return np.where(z > q_es, z / (1.0 - es_alpha), 0.0)

    def var_exact() -> float:
        return q_es

    def var_eval(u: np.ndarray) -> np.ndarray:
        z = ndtri(np.clip(u.reshape(-1), 1e-12, 1 - 1e-12))
        return z

    return {
        "rare_event": MCProblem(
            name="rare_event",
            category="risk",
            description=f"P( standardized mean Z > {rare_threshold} ), d={rare_dim}",
            default_dim=rare_dim,
            exact=rare_exact,
            evaluate=rare_eval,
        ),
        "expected_shortfall": MCProblem(
            name="expected_shortfall",
            category="risk",
            description=f"ES_{es_alpha} of standard normal; exact formula",
            default_dim=1,
            exact=es_exact,
            evaluate=es_eval,
        ),
        "var_quantile": MCProblem(
            name="var_quantile",
            category="risk",
            description=f"VaR_{es_alpha} quantile of standard normal; exact Φ^{-1}(α)",
            default_dim=1,
            exact=var_exact,
            evaluate=var_eval,
            aggregate=lambda z: float(np.quantile(z, es_alpha)),
        ),
    }


def problem_presets() -> dict[str, list[str]]:
    return {
        "finance": [
            "european_call",
            "european_put",
            "digital_call",
            "arithmetic_asian_call",
            "geometric_asian_call",
        ],
        "integral": [
            "exp_sum",
            "indicator_mean",
            "sin_product",
            "poly_sum",
            "gaussian_sphere",
            "g_function",
            "max_coordinate",
            "l2_ball_indicator",
        ],
        "risk": ["rare_event", "expected_shortfall", "var_quantile"],
    }


def resolve_problems(spec: str, catalog: dict[str, MCProblem]) -> list[str]:
    spec = spec.strip().lower()
    if spec == "all":
        return list(catalog.keys())
    presets = problem_presets()
    names: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part in presets:
            names.extend(presets[part])
        elif part in catalog:
            names.append(part)
        else:
            raise ValueError(f"Unknown problem or preset {part!r}")
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def all_problems(**kwargs) -> dict[str, MCProblem]:
    out: dict[str, MCProblem] = {}
    out.update(make_finance_problems(**{k: v for k, v in kwargs.items() if k in (
        "s0", "strike", "rate", "sigma", "maturity", "n_steps", "ref_asian"
    )}))
    out.update(make_integral_problems(dimension=kwargs.get("integral_dim", 12)))
    out.update(make_risk_problems(
        rare_dim=kwargs.get("rare_dim", 16),
        rare_threshold=kwargs.get("rare_threshold", 2.5),
        es_alpha=kwargs.get("es_alpha", 0.99),
    ))
    return out


def fit_alpha(ns: list[int], errs: list[float]) -> float:
    xs, ys = [], []
    for n, e in zip(ns, errs):
        if e > 0:
            xs.append(math.log(float(n)))
            ys.append(math.log(float(e)))
    if len(xs) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.array(xs), np.array(ys), 1)
    return float(-slope)
