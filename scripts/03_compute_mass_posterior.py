#!/usr/bin/env python3
"""
03_compute_mass_posterior.py — Bayesian mass posterior for HD 159254.

The mass function f(M) = 1.51 Msun exceeds the Chandrasekhar
limit but falls below the NS ceiling (~3 Msun), so the BH case
depends critically on the primary mass estimate.

We perform Monte Carlo inclination debiasing with multiple M1
priors spanning the range from cool giant to hot supergiant.

Outputs:
  results/mass_posterior_results.json
  paper/figures/fig_mass_posterior.pdf
"""

import json, os
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── orbital parameters ──────────────────────────────────────────────
P_DAYS = 619.989
P_ERR = 1.449
ECC = 0.00565
ECC_ERR = 0.00615
K1 = 28.623        # km/s
K1_ERR = 0.083

# ─── primary mass ────────────────────────────────────────────────────
# No spectroscopic Teff available. SED analysis will constrain this.
# The star appears very luminous (M_G ~ -4 to -7 depending on A_V).
# Conservative: use M1 = 5.0 ± 3.0 Msun (broad, spans cool giant
# to early B-type). Sensitivity analysis explores wider range.
M1_BEST = 5.0      # Msun
M1_SIGMA = 3.0     # broad to account for large Teff uncertainty

# ─── mass thresholds ─────────────────────────────────────────────────
N_DRAWS = 200_000
NS_MAX = 3.0       # Msun — conservative NS ceiling (Rezzolla+2018)
BH_THRESHOLD = 5.0 # Msun — upper edge of mass gap


def mass_function(P, K1, e):
    """Spectroscopic mass function f(M) in solar masses."""
    return 1.0385e-7 * (1 - e**2)**1.5 * K1**3 * P


def solve_m2_min(fM, M1):
    """Solve M2^3 / (M1+M2)^2 = f(M) for M2 at i=90°."""
    def eq(m2):
        return m2**3 / (M1 + m2)**2 - fM
    if eq(0.01) > 0:
        return 0.01
    return brentq(eq, 0.01, 500.0)


def main():
    print('=== Bayesian Mass Posterior for HD 159254 ===\n')

    # Mass function
    fM = mass_function(P_DAYS, K1, ECC)
    print(f'  f(M) = {fM:.4f} Msun')
    print(f'  f(M) > Chandrasekhar (1.44 Msun): {fM > 1.44}')
    print(f'  f(M) > NS ceiling (3.0 Msun):     {fM > NS_MAX}')
    print(f'  Absolute mass floor (M1->0):       {fM:.2f} Msun')

    # M2_min across M1 range
    m1_values = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    print(f'\n  {"M1":>6} {"M2_min":>8} {"Above NS?":>10} {"Above BH?":>10}')
    for m1 in m1_values:
        m2m = solve_m2_min(fM, m1)
        print(f'  {m1:6.1f} {m2m:8.2f} {"YES" if m2m > NS_MAX else "no":>10}'
              f' {"YES" if m2m > BH_THRESHOLD else "no":>10}')

    m2_fiducial = solve_m2_min(fM, M1_BEST)
    print(f'\n  Fiducial: M1={M1_BEST}, M2_min={m2_fiducial:.2f} Msun')

    # Monte Carlo inclination debiasing
    print(f'\n  Running MC ({N_DRAWS:,} draws) ...')
    rng = np.random.default_rng(42)

    # Draw M1 from lognormal (positive-definite)
    m1_draws = rng.lognormal(
        mean=np.log(M1_BEST) - 0.5 * (M1_SIGMA / M1_BEST)**2,
        sigma=M1_SIGMA / M1_BEST,
        size=N_DRAWS
    )

    # Draw K1 with Gaussian uncertainty
    k1_draws = rng.normal(K1, K1_ERR, N_DRAWS)
    k1_draws = np.clip(k1_draws, 5, 200)

    # Draw orbital elements with Gaussian uncertainty
    p_draws = rng.normal(P_DAYS, P_ERR, N_DRAWS)
    e_draws = np.clip(rng.normal(ECC, ECC_ERR, N_DRAWS), 0, 0.9)

    # Mass function for each draw
    fm_draws = 1.0385e-7 * (1 - e_draws**2)**1.5 * k1_draws**3 * p_draws

    # Isotropic inclinations: P(i) proportional to sin(i)
    cos_i = rng.uniform(0, 1, N_DRAWS)
    sin_i = np.sqrt(1 - cos_i**2)

    m2_draws = []
    for m1, si, fmi in zip(m1_draws, sin_i, fm_draws):
        fM_eff = fmi / si**3
        try:
            m2 = brentq(lambda m2: m2**3 / (m1 + m2)**2 - fM_eff,
                        0.01, 5000.0)
            m2_draws.append(m2)
        except (ValueError, RuntimeError):
            continue

    m2_draws = np.array(m2_draws)
    n_valid = len(m2_draws)
    print(f'  Valid samples: {n_valid:,}/{N_DRAWS:,}')

    # Statistics
    median = np.median(m2_draws)
    ci68 = np.percentile(m2_draws, [16, 84])
    ci90 = np.percentile(m2_draws, [5, 95])
    p_bh = 100 * np.mean(m2_draws > BH_THRESHOLD)
    p_above_ns = 100 * np.mean(m2_draws > NS_MAX)
    p_mg = 100 * np.mean((m2_draws > NS_MAX) & (m2_draws <= BH_THRESHOLD))
    p_ns = 100 * np.mean(m2_draws <= NS_MAX)

    print(f'\n  Results (M1 = {M1_BEST} +/- {M1_SIGMA} Msun):')
    print(f'    M2 median     = {median:.2f} Msun')
    print(f'    68% CI        = [{ci68[0]:.2f}, {ci68[1]:.2f}] Msun')
    print(f'    90% CI        = [{ci90[0]:.2f}, {ci90[1]:.2f}] Msun')
    print(f'    P(>NS ceiling) = {p_above_ns:.1f}%')
    print(f'    P(BH > 5)     = {p_bh:.1f}%')
    print(f'    P(mass gap)   = {p_mg:.1f}%')
    print(f'    P(NS)         = {p_ns:.1f}%')

    # Save results
    basedir = os.path.dirname(__file__)
    results = {
        'mass_function_msun': round(fM, 4),
        'M1_best': M1_BEST,
        'M1_sigma': M1_SIGMA,
        'M2_min_fiducial': round(m2_fiducial, 2),
        'M2_absolute_floor': round(fM, 2),
        'MC_draws': N_DRAWS,
        'MC_valid': n_valid,
        'M2_median': round(float(median), 2),
        'M2_68ci': [round(float(ci68[0]), 2), round(float(ci68[1]), 2)],
        'M2_90ci': [round(float(ci90[0]), 2), round(float(ci90[1]), 2)],
        'P_above_NS_percent': round(float(p_above_ns), 1),
        'P_BH_percent': round(float(p_bh), 1),
        'P_massgap_percent': round(float(p_mg), 1),
        'P_NS_percent': round(float(p_ns), 1),
        'M2_min_table': {str(m1): round(solve_m2_min(fM, m1), 2)
                         for m1 in m1_values},
    }
    outpath = os.path.join(basedir, '..', 'results',
                           'mass_posterior_results.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Saved: {outpath}')

    # Figure
    figpath = os.path.join(basedir, '..', 'paper', 'figures',
                           'fig_mass_posterior.pdf')
    os.makedirs(os.path.dirname(figpath), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: M2_min vs M1
    m1_grid = np.linspace(0.3, 20, 300)
    m2min_grid = [solve_m2_min(fM, m1) for m1 in m1_grid]
    ax1.plot(m1_grid, m2min_grid, 'b-', lw=2)
    ax1.axhline(BH_THRESHOLD, color='red', ls='--', alpha=0.7,
                label=r'$M_2 = 5\,M_\odot$ (BH threshold)')
    ax1.axhline(NS_MAX, color='orange', ls='--', alpha=0.7,
                label=r'$M_2 = 3\,M_\odot$ (NS ceiling)')
    ax1.axhline(fM, color='gray', ls=':', alpha=0.5,
                label=f'$f(M) = {fM:.2f}\\,M_\\odot$ (absolute floor)')
    ax1.axvspan(max(0.3, M1_BEST - M1_SIGMA),
                M1_BEST + M1_SIGMA, alpha=0.15, color='blue',
                label=f'$M_1 = {M1_BEST:.1f} \\pm {M1_SIGMA:.1f}$')
    ax1.axvline(M1_BEST, color='blue', ls=':', alpha=0.5)
    ax1.set_xlabel(r'$M_1$ ($M_\odot$)', fontsize=12)
    ax1.set_ylabel(r'$M_{2,\mathrm{min}}$ ($M_\odot$)', fontsize=12)
    ax1.set_title(r'Minimum companion mass vs $M_1$')
    ax1.legend(fontsize=7, loc='upper left')
    ax1.set_xlim(0.3, 20)
    ax1.set_ylim(1, 16)

    # Right: mass posterior histogram
    bins = np.linspace(1, 60, 120)
    ax2.hist(m2_draws[m2_draws < 60], bins=bins, density=True,
             color='steelblue', alpha=0.7, edgecolor='navy', lw=0.3)
    ax2.axvline(BH_THRESHOLD, color='red', ls='--', lw=2,
                label=f'BH threshold (5 $M_\\odot$)')
    ax2.axvline(NS_MAX, color='orange', ls='--', lw=1.5,
                label=f'NS ceiling (3 $M_\\odot$)')
    ax2.axvline(median, color='black', ls='-', lw=2,
                label=f'Median = {median:.1f} $M_\\odot$')
    ax2.axvspan(ci68[0], ci68[1], alpha=0.15, color='green',
                label=f'68% CI [{ci68[0]:.1f}, {ci68[1]:.1f}]')
    ax2.text(0.97, 0.95, f'$P(\\mathrm{{BH}}) = {p_bh:.1f}\\%$',
             transform=ax2.transAxes, ha='right', va='top', fontsize=14,
             bbox=dict(boxstyle='round', fc='lightyellow'))
    ax2.set_xlabel(r'$M_2$ ($M_\odot$)', fontsize=12)
    ax2.set_ylabel('Probability density', fontsize=12)
    ax2.set_title(f'Inclination-debiased mass posterior ({N_DRAWS//1000}K draws)')
    ax2.legend(fontsize=7)
    ax2.set_xlim(1, 60)

    fig.suptitle('HD 159254 — Mass Constraints', fontsize=13,
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(figpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  -> {figpath}')

    print('\n=== Mass posterior complete ===')


if __name__ == '__main__':
    main()
