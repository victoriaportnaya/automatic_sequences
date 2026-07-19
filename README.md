# Noncorrelated Automatic Sequences and Monte Carlo Integration

This repository contains the principal artifacts and reproducibility files for Victoria Portnaya's Experimental Mathematics project at the Kyiv School of Economics, supervised by Jakub Konieczny.

The project studies automatic sequences obtained by counting binary patterns, their limiting autocorrelations, and their use in Monte Carlo and quasi-Monte Carlo integration. Rudin-Shapiro is treated theoretically, while finite-state computations test larger finite catalogs.

## Main artifacts

- [`artifacts/report/automatic_sequences_monte_carlo_report.pdf`](artifacts/report/automatic_sequences_monte_carlo_report.pdf): final human-readable research report.
- [`artifacts/presentation/nc_monte_carlo_presentation.pdf`](artifacts/presentation/nc_monte_carlo_presentation.pdf): 15-minute presentation.
- [`artifacts/figures/`](artifacts/figures/): figures used in the report.
- [`data/`](data/): principal saved results in JSON and CSV form.
- [`src/`](src/): scripts used for sequence generation, exact-recursion experiments, Fourier diagnostics, and Monte Carlo validation.

## Mathematical scope

For lag \(h\), the theoretical finite autocorrelation is

\[
\gamma_N(h)=\frac{1}{N}\sum_{n=0}^{N-1}a(n)\overline{a(n+h)}.
\]

The direct-prefix programs use the overlap-prefix statistic

\[
\widehat\gamma_N(h)=\frac{1}{N}\sum_{n=0}^{N-h-1}a(n)\overline{a(n+h)}.
\]

For a fixed lag, these have the same limit because

\[
\left|\gamma_N(h)-\widehat\gamma_N(h)\right|\leq \frac{h}{N}.
\]

The prefix-convergence experiment uses lags 1 through 32. The finite-state censuses classify cases on lags 1 through 64 with tolerance 0.005. Except for results proved separately, such labels are operational finite-window classifications rather than all-lag proofs.

## Principal findings

- Rudin-Shapiro is proved to have zero limiting autocorrelation at every nonzero lag.
- The two finite censuses contain 846 and 2,047 pattern sets and reveal operationally NC cases primarily at even moduli.
- Directly packing automatic-sequence bits into multidimensional points is unreliable.
- Using the same sequences as digital shifts of a Sobol net preserves the net structure and performs much better.
- In the paired benchmark, NC-derived shifts and uniform random digital shifts are effectively tied: the geometric-mean final-RMSE ratio is 0.9894.

## Reproduction

The scripts are intended to be run from the repository root. The principal entry points are:

```bash
python3 src/nc_one_over_n_convergence.py
python3 src/focused_mc_validation.py --out data/focused_mc_validation.json
MPLBACKEND=Agg python3 src/make_focused_mc_figures.py
```

The saved data are included so that the main numerical claims can be checked without rerunning the largest censuses.

## Contents policy

This publication repository intentionally excludes LaTeX source files, temporary build files, grading exports, and obsolete report variants.
