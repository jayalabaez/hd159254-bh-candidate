#!/usr/bin/env python3
"""
07_sensitivity_analysis.py — P(BH) sensitivity to prior assumptions.

HD 159254 has f(M) = 1.51 Msun (below NS ceiling), making the BH
classification dependent on M1.  This script explores P(BH) across
a wide range of M1 priors, BH thresholds, and NS ceilings.

Outputs:
  results/sensitivity_results.json
"""

import json, os
import numpy as np
from scipy.optimize import brentq

# ─── Fixed orbital parameters ────────────────────────────────────────
P_DAYS = 619.989
ECC = 0.00565
K1 = 28.623  # km/s

N_DRAWS = 50_000
SEED = 42


def mass_function(P, K1, e):
    return 1.0385e-7 * (1 - e**2)**1.5 * K1**3 * P


def run_mc(M1_best, M1_sigma, bh_threshold, fM, seed=SEED):
    """Run MC and return P(BH) + statistics."""
    rng = np.random.default_rng(seed)
    m1_draws = rng.lognormal(
        mean=np.log(M1_best) - 0.5 * (M1_sigma / M1_best)**2,
        sigma=M1_sigma / M1_best,
        size=N_DRAWS
    )
    cos_i = rng.uniform(0, 1, N_DRAWS)
    sin_i = np.sqrt(1 - cos_i**2)

    m2_draws = []
    for m1, si in zip(m1_draws, sin_i):
        fM_eff = fM / si**3
        try:
            m2 = brentq(lambda m2: m2**3 / (m1 + m2)**2 - fM_eff,
                        0.01, 5000.0)
            m2_draws.append(m2)
        except (ValueError, RuntimeError):
            continue

    m2_arr = np.array(m2_draws)
    p_bh = 100.0 * np.mean(m2_arr > bh_threshold)
    p_ns = 100.0 * np.mean(m2_arr <= 3.0)
    p_mg = 100.0 * np.mean((m2_arr > 3.0) & (m2_arr <= bh_threshold))
    median = float(np.median(m2_arr))
    ci68 = [float(np.percentile(m2_arr, 16)),
            float(np.percentile(m2_arr, 84))]
    return round(p_bh, 1), round(median, 1), [round(c, 1) for c in ci68], round(p_ns, 1), round(p_mg, 1)


def main():
    fM = mass_function(P_DAYS, K1, ECC)
    print(f'f(M) = {fM:.4f} Msun\n')
    print(f'f(M) > Chandrasekhar (1.44): {fM > 1.44}')
    print(f'f(M) > NS ceiling (3.0):     {fM > 3.0}\n')

    configs = []

    # 1. Conservative cool giant: M1 = 2.0 ± 1.0
    for bh_thr in [3.0, 5.0]:
        label = f'BH > {bh_thr:.0f} Msun, M1=2.0+/-1.0 (cool giant)'
        p, med, ci, pns, pmg = run_mc(2.0, 1.0, bh_thr, fM)
        configs.append({
            'label': label, 'BH_threshold': bh_thr,
            'M1': 2.0, 'M1_sigma': 1.0,
            'P_BH': p, 'P_NS': pns, 'P_massgap': pmg,
            'M2_median': med, 'M2_68ci': ci,
        })

    # 2. Moderate giant: M1 = 3.0 ± 1.5
    for bh_thr in [3.0, 5.0]:
        label = f'BH > {bh_thr:.0f} Msun, M1=3.0+/-1.5 (moderate giant)'
        p, med, ci, pns, pmg = run_mc(3.0, 1.5, bh_thr, fM)
        configs.append({
            'label': label, 'BH_threshold': bh_thr,
            'M1': 3.0, 'M1_sigma': 1.5,
            'P_BH': p, 'P_NS': pns, 'P_massgap': pmg,
            'M2_median': med, 'M2_68ci': ci,
        })

    # 3. Fiducial: M1 = 5.0 ± 3.0
    for bh_thr in [3.0, 5.0]:
        label = f'BH > {bh_thr:.0f} Msun, M1=5.0+/-3.0 (fiducial)'
        p, med, ci, pns, pmg = run_mc(5.0, 3.0, bh_thr, fM)
        configs.append({
            'label': label, 'BH_threshold': bh_thr,
            'M1': 5.0, 'M1_sigma': 3.0,
            'P_BH': p, 'P_NS': pns, 'P_massgap': pmg,
            'M2_median': med, 'M2_68ci': ci,
        })

    # 4. Luminous star: M1 = 8.0 ± 3.0
    label = 'BH > 5 Msun, M1=8.0+/-3.0 (luminous star)'
    p, med, ci, pns, pmg = run_mc(8.0, 3.0, 5.0, fM)
    configs.append({
        'label': label, 'BH_threshold': 5.0,
        'M1': 8.0, 'M1_sigma': 3.0,
        'P_BH': p, 'P_NS': pns, 'P_massgap': pmg,
        'M2_median': med, 'M2_68ci': ci,
    })

    # 5. Hot supergiant: M1 = 12.0 ± 4.0
    label = 'BH > 5 Msun, M1=12.0+/-4.0 (B supergiant)'
    p, med, ci, pns, pmg = run_mc(12.0, 4.0, 5.0, fM)
    configs.append({
        'label': label, 'BH_threshold': 5.0,
        'M1': 12.0, 'M1_sigma': 4.0,
        'P_BH': p, 'P_NS': pns, 'P_massgap': pmg,
        'M2_median': med, 'M2_68ci': ci,
    })

    # 6. Extreme high: M1 = 15.0 ± 5.0 (pipeline estimate)
    label = 'BH > 5 Msun, M1=15.0+/-5.0 (pipeline est.)'
    p, med, ci, pns, pmg = run_mc(15.0, 5.0, 5.0, fM)
    configs.append({
        'label': label, 'BH_threshold': 5.0,
        'M1': 15.0, 'M1_sigma': 5.0,
        'P_BH': p, 'P_NS': pns, 'P_massgap': pmg,
        'M2_median': med, 'M2_68ci': ci,
    })

    # 7. Extreme low: M1 = 1.0 ± 0.5
    label = 'BH > 5 Msun, M1=1.0+/-0.5 (extreme low)'
    p, med, ci, pns, pmg = run_mc(1.0, 0.5, 5.0, fM)
    configs.append({
        'label': label, 'BH_threshold': 5.0,
        'M1': 1.0, 'M1_sigma': 0.5,
        'P_BH': p, 'P_NS': pns, 'P_massgap': pmg,
        'M2_median': med, 'M2_68ci': ci,
    })

    # Print table
    print(f'{"Configuration":<52} {"P(BH)":>7} {"P(NS)":>7} {"P(MG)":>7} '
          f'{"M2 med":>7} {"68% CI":>15}')
    print('-' * 100)
    for c in configs:
        ci_str = f'[{c["M2_68ci"][0]:.1f}, {c["M2_68ci"][1]:.1f}]'
        print(f'{c["label"]:<52} {c["P_BH"]:>6.1f}% {c["P_NS"]:>6.1f}% '
              f'{c["P_massgap"]:>6.1f}% {c["M2_median"]:>6.1f} {ci_str:>15}')

    # Save
    basedir = os.path.dirname(__file__)
    outpath = os.path.join(basedir, '..', 'results',
                           'sensitivity_results.json')
    with open(outpath, 'w') as f:
        json.dump(configs, f, indent=2)
    print(f'\nSaved: {outpath}')


if __name__ == '__main__':
    main()
