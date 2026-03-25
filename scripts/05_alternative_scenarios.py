#!/usr/bin/env python3
"""
05_alternative_scenarios.py — Systematic test of non-BH explanations
for the dark companion in HD 159254.

Unlike CPD-67 2116A (where f(M) > NS ceiling), HD 159254 has
f(M) = 1.51 Msun — above the Chandrasekhar limit but below the
NS ceiling.  The BH case therefore depends on the primary mass.

Tests six scenarios:
  1. Main-sequence star  → depends on primary luminosity
  2. White dwarf          → EXCLUDED (f(M) > Chandrasekhar)
  3. Neutron star         → depends on M1
  4. Hierarchical triple  → UNLIKELY (stability + photometry)
  5. Stripped He star     → DISFAVOURED (needs UV check)
  6. Astrometric artefact → UNLIKELY (σ=343, but RUWE=2.32)

Outputs:
  results/alternative_scenarios_results.json
"""

import json, os
import numpy as np
from scipy.optimize import brentq

# ── Constants ─────────────────────────────────────────────────────────
M_CHANDRASEKHAR = 1.44   # Msun
M_TOV = 2.3              # Msun — conservative NS ceiling
P_ORBIT = 619.989        # days
E_ORBIT = 0.00565
K1 = 28.623              # km/s
RUWE = 2.32
SIG = 343.4              # NSS solution significance
L_PRIMARY = 500.0        # Lsun (conservative; updated after SED)
TEFF_PRIMARY = 5000      # K (placeholder; updated after SED)

# Compute mass function
FM = 1.0385e-7 * (1 - E_ORBIT**2)**1.5 * K1**3 * P_ORBIT

# Fiducial primary mass and resulting M2_min
M1 = 5.0                # Msun
M2_MIN = brentq(lambda m2: m2**3 / (M1 + m2)**2 - FM, 0.01, 500.0)


def try_update_from_results():
    """Update parameters from previous analysis results."""
    global L_PRIMARY, TEFF_PRIMARY, M1, M2_MIN
    basedir = os.path.dirname(__file__)
    sed_path = os.path.join(basedir, '..', 'results', 'sed_fit_results.json')
    if os.path.exists(sed_path):
        with open(sed_path) as f:
            sed = json.load(f)
        TEFF_PRIMARY = sed.get('best_Teff_K', TEFF_PRIMARY)
        mg = sed.get('M_G_corrected', -4.0)
        bc = -0.2
        mbol = mg + bc
        log_l = (4.74 - mbol) / 2.5
        L_PRIMARY = max(10**log_l, 100)
        M1 = sed.get('M1_estimate', M1)
        M2_MIN = brentq(lambda m2: m2**3 / (M1 + m2)**2 - FM, 0.01, 500.0)


def ms_luminosity(M):
    if M < 0.45: return 10**(2.028 * np.log10(M) - 0.976)
    elif M < 2.0: return M**4.572
    elif M < 7.0: return 10**(3.962 * np.log10(M) + 0.120)
    else: return 10**(2.726 * np.log10(M) + 1.237)


def test_ms_companion():
    l_comp = ms_luminosity(M2_MIN)
    flux_ratio_pct = l_comp / L_PRIMARY * 100
    detectable = flux_ratio_pct > 1
    return {
        'scenario': 'Main-sequence companion',
        'test': 'Photometric flux ratio test',
        'M2_required': round(M2_MIN, 1),
        'expected_L': round(l_comp, 0),
        'flux_ratio_pct': round(flux_ratio_pct, 1),
        'verdict': 'EXCLUDED' if detectable else 'NOT EXCLUDED',
        'reason': (f'A {M2_MIN:.1f} Msun MS star would contribute '
                   f'~{flux_ratio_pct:.1f}% of primary flux. '
                   + ('This is detectable; no secondary SED is seen.'
                      if detectable else
                      'This may be below detection threshold for a '
                      'luminous primary.')),
    }


def test_white_dwarf():
    return {
        'scenario': 'White dwarf',
        'test': 'Mass ceiling (Chandrasekhar)',
        'M_chandrasekhar': M_CHANDRASEKHAR,
        'mass_function': round(FM, 4),
        'verdict': 'EXCLUDED',
        'reason': (f'f(M) = {FM:.2f} Msun exceeds the Chandrasekhar '
                   f'limit ({M_CHANDRASEKHAR} Msun) by a factor of '
                   f'{FM/M_CHANDRASEKHAR:.2f}. No WD can have this mass.'),
    }


def test_neutron_star():
    # f(M) = 1.51 < 3.0, so NS not excluded by dynamics alone
    # But M2_min at M1=5 is ~5.5, which IS above NS ceiling
    above_ns = M2_MIN > 3.0
    return {
        'scenario': 'Neutron star',
        'test': 'Mass ceiling (TOV limit) + primary mass',
        'M_tov': M_TOV,
        'mass_function': round(FM, 4),
        'M1_fiducial': M1,
        'M2_min_at_fiducial': round(M2_MIN, 2),
        'verdict': 'EXCLUDED' if above_ns else 'NOT EXCLUDED',
        'reason': (f'f(M) = {FM:.2f} Msun is below the NS ceiling '
                   f'({M_TOV} Msun), so a NS is NOT excluded by the '
                   f'mass function alone. However, at fiducial '
                   f'M1 = {M1:.1f} Msun, M2_min = {M2_MIN:.1f} Msun '
                   f'{"exceeds" if above_ns else "may not exceed"} '
                   f'the NS ceiling. '
                   f'The NS scenario depends on the primary mass estimate.'),
    }


def test_hierarchical_triple():
    P_inner_max = P_ORBIT / 4.7 * (1 - E_ORBIT)**1.8
    M_each = M2_MIN / 2
    l_each = ms_luminosity(M_each)
    return {
        'scenario': 'Hierarchical triple',
        'test': 'Mardling-Aarseth stability + photometric test',
        'P_outer_d': P_ORBIT,
        'P_inner_max_d': round(P_inner_max, 1),
        'M2_split': [round(M_each, 1), round(M_each, 1)],
        'L_each': round(l_each, 0),
        'verdict': 'UNLIKELY',
        'reason': (f'Stability requires P_inner < {P_inner_max:.1f} d. '
                   f'Two ~{M_each:.1f} Msun MS stars would each have '
                   f'L ~ {l_each:.0f} Lsun. If both are compact objects, '
                   f'the inner binary would need M_total > {M2_MIN:.1f} '
                   f'Msun — still requiring at least one BH-mass object. '
                   f'Near-zero eccentricity (e={E_ORBIT:.3f}) also '
                   f'disfavours a triple system where Kozai-Lidov '
                   f'oscillations would pump eccentricity.'),
    }


def test_stripped_star():
    return {
        'scenario': 'Stripped helium star',
        'test': 'UV excess + mass requirement',
        'M2_required': round(M2_MIN, 1),
        'expected_Teff': '> 30000 K',
        'verdict': 'DISFAVOURED',
        'reason': (f'A {M2_MIN:.1f} Msun stripped star would have '
                   f'Teff > 30000 K with strong UV excess and He II '
                   f'emission. No UV data available. Cannot be fully '
                   f'excluded without UV spectroscopy.'),
    }


def test_artefact():
    return {
        'scenario': 'Astrometric/pipeline artefact',
        'test': 'NSS solution quality + RUWE',
        'significance': SIG,
        'RUWE': RUWE,
        'verdict': 'UNLIKELY',
        'reason': (f'The NSS significance sigma = {SIG:.1f} is very high '
                   f'(68x the catalogue threshold of 5). The orbital '
                   f'elements (P={P_ORBIT:.1f}d, e={E_ORBIT:.3f}) are '
                   f'physically consistent. However, RUWE = {RUWE:.2f} '
                   f'exceeds the 1.4 threshold, indicating astrometric '
                   f'residuals beyond a single-star model. The elevated '
                   f'RUWE may arise from the binary orbital motion itself '
                   f'or from the close neighbor at 3.3 arcsec. Independent '
                   f'RV confirmation is essential to exclude systematics.'),
    }


def main():
    print('=== Alternative Scenario Analysis for HD 159254 ===\n')

    try_update_from_results()

    print(f'  f(M) = {FM:.4f} Msun')
    print(f'  M1 = {M1:.1f} Msun (fiducial)')
    print(f'  M2_min = {M2_MIN:.2f} Msun')
    print()

    tests = [
        test_ms_companion(),
        test_white_dwarf(),
        test_neutron_star(),
        test_hierarchical_triple(),
        test_stripped_star(),
        test_artefact(),
    ]

    n_excluded = sum(1 for t in tests if t['verdict'] == 'EXCLUDED')
    n_unlikely = sum(1 for t in tests if t['verdict'] == 'UNLIKELY')
    n_disfavoured = sum(1 for t in tests if t['verdict'] == 'DISFAVOURED')
    n_open = sum(1 for t in tests if t['verdict'] == 'NOT EXCLUDED')

    for t in tests:
        print(f'  [{t["verdict"]:16s}] {t["scenario"]}')
        print(f'    {t["reason"]}\n')

    print(f'  Summary: {n_excluded} EXCLUDED, {n_unlikely} UNLIKELY, '
          f'{n_disfavoured} DISFAVOURED, {n_open} NOT EXCLUDED '
          f'out of {len(tests)} scenarios')

    # Save
    basedir = os.path.dirname(__file__)
    results = {
        'mass_function': round(FM, 4),
        'M1_fiducial': M1,
        'M2_min_fiducial': round(M2_MIN, 2),
        'scenarios': tests,
        'summary': {
            'excluded': n_excluded,
            'unlikely': n_unlikely,
            'disfavoured': n_disfavoured,
            'not_excluded': n_open,
            'total': len(tests),
        },
        'key_argument': (
            f'f(M) = {FM:.2f} Msun exceeds the Chandrasekhar limit '
            f'but falls below the NS ceiling ({M_TOV} Msun). The BH '
            f'case depends on the primary mass: at fiducial '
            f'M1 = {M1:.1f} Msun, M2_min = {M2_MIN:.1f} Msun '
            f'enters BH territory. The very high NSS significance '
            f'(sigma = {SIG:.1f}) and physically coherent orbital '
            f'elements support a genuine binary.'
        ),
    }
    outpath = os.path.join(basedir, '..', 'results',
                           'alternative_scenarios_results.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Saved: {outpath}')

    print('\n=== Alternative scenario analysis complete ===')


if __name__ == '__main__':
    main()
