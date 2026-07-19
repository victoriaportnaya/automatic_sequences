# Noncorrelated Automatic Sequences and Monte Carlo Integration

This repository contains the artifacts and files for Experimental Mathematics project at the Kyiv School of Economics.

The project studies automatic sequences obtained by counting binary patterns, their limiting autocorrelations, and their use in Monte Carlo and quasi-Monte Carlo integration. 

## Main artifacts
- [`artifacts/figures/`](artifacts/figures/): figures used in the report.
- [`data/`](data/): principal saved results in JSON and CSV form.
- [`src/`](src/): scripts used for sequence generation, exact-recursion experiments, Fourier diagnostics, and Monte Carlo validation.


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

