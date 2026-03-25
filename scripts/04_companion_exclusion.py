#!/usr/bin/env python3
"""
04_companion_exclusion.py — Luminous companion exclusion test for HD 159254.

Demonstrates that a main-sequence star massive enough to explain the
orbital dynamics would contribute significant flux — and assesses
detectability.

Outputs:
  results/companion_exclusion_results.json
  paper/figures/fig_companion_exclusion.pdf
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── constants ────────────────────────────────────────────────────────
h_cgs = 6.626e-27
c_cgs = 2.998e10
k_cgs = 1.381e-16
Jy = 1e-23
Lsun = 3.828e33   # erg/s
Rsun = 6.957e10   # cm
sigma_sb = 5.670e-5

# ─── HD 159254 primary ───────────────────────────────────────────────
# Estimated from SED: luminous evolved star.
# Conservative estimates (to be updated after running 02):
# M_G ~ -4 to -7 depending on extinction; L ~ 500–10000 Lsun
# Use conservative lower-bound to get LOWER-bound on flux ratio
TEFF_PRIMARY = 5000    # K (placeholder; updated after SED run)
L_PRIMARY = 500.0      # Lsun (conservative lower bound)
M1 = 5.0               # Msun (fiducial)
M2_MIN = 5.5           # Msun (from mass posterior at M1=5.0)

FILTERS = {
    'G':  {'lam': 0.622, 'fzp': 3228.75},
    'BP': {'lam': 0.511, 'fzp': 3552.01},
    'RP': {'lam': 0.777, 'fzp': 2554.95},
    'J':  {'lam': 1.235, 'fzp': 1594.0},
    'H':  {'lam': 1.662, 'fzp': 1024.0},
    'Ks': {'lam': 2.159, 'fzp': 666.7},
    'W1': {'lam': 3.353, 'fzp': 309.5},
    'W2': {'lam': 4.603, 'fzp': 171.8},
}
DETECTION_THRESHOLD = 0.01  # 1% of primary flux


def ms_luminosity(M):
    """Main-sequence luminosity from Eker+2018."""
    if M < 0.45:
        return 10**(2.028 * np.log10(M) - 0.976)
    elif M < 2.0:
        return M**4.572
    elif M < 7.0:
        return 10**(3.962 * np.log10(M) + 0.120)
    else:
        return 10**(2.726 * np.log10(M) + 1.237)


def ms_teff(M):
    """Approximate MS Teff from mass (empirical)."""
    return 5778 * M**0.57


def ms_radius(M):
    """MS radius from L and Teff."""
    L = ms_luminosity(M) * Lsun
    T = ms_teff(M)
    return np.sqrt(L / (4 * np.pi * sigma_sb * T**4)) / Rsun


def planck_ratio(T1, T2, lam_um):
    """Flux ratio B_ν(T2)/B_ν(T1)."""
    lam_cm = lam_um * 1e-4
    nu = c_cgs / lam_cm
    x1 = h_cgs * nu / (k_cgs * T1)
    x2 = h_cgs * nu / (k_cgs * T2)
    if x1 > 500 or x2 > 500:
        return 0.0
    B1 = 1.0 / (np.exp(x1) - 1)
    B2 = 1.0 / (np.exp(x2) - 1)
    return B2 / B1 if B1 > 0 else 0.0


def compute_flux_ratios(m_comp, teff_prim, l_prim):
    """Companion/primary flux ratios in each band."""
    l_comp = ms_luminosity(m_comp)
    t_comp = ms_teff(m_comp)
    bol_ratio = l_comp / l_prim

    ratios = {}
    for band, filt in FILTERS.items():
        r_planck = planck_ratio(teff_prim, t_comp, filt['lam'])
        ratios[band] = bol_ratio * r_planck

    return ratios, l_comp, t_comp, bol_ratio


def find_max_hidden_mass(teff_prim, l_prim, threshold=0.01):
    """Find max companion mass hidden below detection threshold."""
    for m in np.arange(0.1, 30.0, 0.01):
        ratios, _, _, _ = compute_flux_ratios(m, teff_prim, l_prim)
        if any(r > threshold for r in ratios.values()):
            return round(m - 0.01, 2)
    return 30.0


def try_update_from_sed():
    """Try to read SED results and update parameters."""
    global TEFF_PRIMARY, L_PRIMARY, M1, M2_MIN
    basedir = os.path.dirname(__file__)
    sed_path = os.path.join(basedir, '..', 'results', 'sed_fit_results.json')
    mp_path = os.path.join(basedir, '..', 'results', 'mass_posterior_results.json')

    if os.path.exists(sed_path):
        with open(sed_path) as f:
            sed = json.load(f)
        TEFF_PRIMARY = sed.get('best_Teff_K', TEFF_PRIMARY)
        mg = sed.get('M_G_corrected', -4.0)
        # Estimate L from M_G
        bc = -0.2  # approximate
        mbol = mg + bc
        log_l = (4.74 - mbol) / 2.5
        L_PRIMARY = max(10**log_l, 100)
        M1 = sed.get('M1_estimate', M1)
        print(f'  Updated from SED: Teff={TEFF_PRIMARY}K, L={L_PRIMARY:.0f}Lsun, M1={M1:.1f}')

    if os.path.exists(mp_path):
        with open(mp_path) as f:
            mp = json.load(f)
        M2_MIN = mp.get('M2_min_fiducial', M2_MIN)
        print(f'  Updated from mass posterior: M2_min={M2_MIN:.1f}')

    from scipy.optimize import brentq
    fM = 1.0385e-7 * (1 - 0.00565**2)**1.5 * 28.623**3 * 619.989
    M2_MIN = brentq(lambda m2: m2**3 / (M1 + m2)**2 - fM, 0.01, 500.0)


def main():
    print('=== Companion Exclusion Test for HD 159254 ===\n')

    try_update_from_sed()

    ratios, l_comp, t_comp, bol_ratio = compute_flux_ratios(
        M2_MIN, TEFF_PRIMARY, L_PRIMARY)
    r_comp = ms_radius(M2_MIN)

    print(f'  Primary:   M1~{M1:.1f} Msun, Teff~{TEFF_PRIMARY} K, '
          f'L~{L_PRIMARY:.0f} Lsun')
    print(f'  Companion: M2_min={M2_MIN:.1f} Msun (hypothetical MS)')
    print(f'    L_comp   = {l_comp:.1f} Lsun')
    print(f'    Teff_comp= {t_comp:.0f} K')
    print(f'    R_comp   = {r_comp:.2f} Rsun')
    print(f'    Bol ratio= {bol_ratio:.4f} ({bol_ratio*100:.1f}%)')
    print(f'    Brighten = {-2.5*np.log10(1+bol_ratio):.3f} mag')

    print(f'\n  Band-by-band flux ratios (F_comp/F_prim):')
    for band, ratio in ratios.items():
        status = 'EASILY DETECTABLE' if ratio > 0.1 else \
                 'DETECTABLE' if ratio > DETECTION_THRESHOLD else 'HIDDEN'
        print(f'    {band:4s}: {ratio:.4f} ({ratio*100:.1f}%) -> {status}')

    max_hidden = find_max_hidden_mass(TEFF_PRIMARY, L_PRIMARY)
    print(f'\n  Maximum MS mass hidden in photometry: {max_hidden} Msun')
    print(f'  M2_min exceeds max-hidden by {M2_MIN - max_hidden:.1f} Msun')

    verdict = ('LUMINOUS COMPANION EXCLUDED' if max_hidden < M2_MIN
               else 'COMPANION MAY BE HIDDEN')

    # Save results
    basedir = os.path.dirname(__file__)
    results = {
        'M1': M1, 'M2_min': round(M2_MIN, 2),
        'L_primary': round(L_PRIMARY, 1), 'Teff_primary': TEFF_PRIMARY,
        'L_companion_hyp': round(l_comp, 1),
        'Teff_companion_hyp': round(t_comp, 0),
        'R_companion_hyp': round(r_comp, 2),
        'bolometric_ratio': round(bol_ratio, 4),
        'brightening_mag': round(-2.5 * np.log10(1 + bol_ratio), 3),
        'band_flux_ratios': {b: round(r, 4) for b, r in ratios.items()},
        'max_hidden_mass': max_hidden,
        'detection_threshold': DETECTION_THRESHOLD,
        'verdict': verdict,
    }
    outpath = os.path.join(basedir, '..', 'results',
                           'companion_exclusion_results.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Saved: {outpath}')

    # Figure
    figpath = os.path.join(basedir, '..', 'paper', 'figures',
                           'fig_companion_exclusion.pdf')
    os.makedirs(os.path.dirname(figpath), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: flux ratio per band
    band_names = list(ratios.keys())
    band_ratios = [ratios[b] * 100 for b in band_names]

    ax1.bar(range(len(band_names)), band_ratios, color='indianred',
            edgecolor='darkred', alpha=0.8)
    ax1.axhline(DETECTION_THRESHOLD * 100, color='green', ls='--', lw=2,
                label=f'Detection threshold ({DETECTION_THRESHOLD*100:.0f}%)')
    ax1.set_xticks(range(len(band_names)))
    ax1.set_xticklabels(band_names, fontsize=9)
    ax1.set_ylabel(r'$F_\mathrm{comp} / F_\mathrm{prim}$ (%)', fontsize=11)
    ax1.set_title(f'Companion flux if $M_2 = {M2_MIN:.1f}\\,M_\\odot$ (MS)')
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, max(band_ratios) * 1.2 if max(band_ratios) > 0 else 1)

    # Right: M2 vs maximum detection
    m_grid = np.arange(0.5, 15, 0.1)
    bol_grid = [ms_luminosity(m) / L_PRIMARY for m in m_grid]
    ax2.plot(m_grid, bol_grid, 'b-', lw=2, label='Bol. flux ratio')
    ax2.axhline(DETECTION_THRESHOLD, color='green', ls='--',
                label=f'Detection threshold (1%)')
    ax2.axvline(M2_MIN, color='red', ls='--',
                label=f'$M_{{2,min}} = {M2_MIN:.1f}\\,M_\\odot$')
    ax2.axvline(max_hidden, color='orange', ls=':',
                label=f'Max hidden ({max_hidden} $M_\\odot$)')
    ax2.set_xlabel(r'Companion mass ($M_\odot$)', fontsize=11)
    ax2.set_ylabel('Bolometric flux ratio', fontsize=11)
    ax2.set_title('Companion detectability vs mass')
    ax2.set_yscale('log')
    ax2.legend(fontsize=8)
    ax2.set_xlim(0.5, 15)

    fig.suptitle('HD 159254 — Companion Exclusion', fontsize=13,
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(figpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  -> {figpath}')

    print('\n=== Companion exclusion complete ===')


if __name__ == '__main__':
    main()
