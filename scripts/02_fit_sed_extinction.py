#!/usr/bin/env python3
"""
02_fit_sed_extinction.py — SED analysis with extinction for HD 159254.

No independent Teff is available (no GSP-Phot, no spectroscopic
survey match).  We perform a blackbody grid search across a wide
Teff range (3500–25000 K) using Gaia BP-RP colour to self-
consistently constrain (Teff, A_V).

The target lies toward the Galactic Centre (l=0.59, b=+3.20)
where significant interstellar extinction is expected.

Outputs:
  results/sed_fit_results.json
  paper/figures/fig_sed_extinction.pdf
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ─── constants ────────────────────────────────────────────────────────
h_cgs = 6.626e-27     # erg s
c_cgs = 2.998e10      # cm/s
k_cgs = 1.381e-16     # erg/K
Jy = 1e-23            # erg/s/cm²/Hz

# Filter data: zero-point flux (Jy), effective wavelength (μm),
# A(λ)/A_V from Cardelli+1989 / Wang & Chen 2019
FILTERS = {
    'B':  {'fzp': 4130.0,  'lam': 0.440, 'aav': 1.337},
    'G':  {'fzp': 3228.75, 'lam': 0.622, 'aav': 0.789},
    'BP': {'fzp': 3552.01, 'lam': 0.511, 'aav': 1.002},
    'RP': {'fzp': 2554.95, 'lam': 0.777, 'aav': 0.589},
    'J':  {'fzp': 1594.0,  'lam': 1.235, 'aav': 0.282},
    'H':  {'fzp': 1024.0,  'lam': 1.662, 'aav': 0.175},
    'Ks': {'fzp': 666.7,   'lam': 2.159, 'aav': 0.112},
    'W1': {'fzp': 309.5,   'lam': 3.353, 'aav': 0.065},
    'W2': {'fzp': 171.8,   'lam': 4.603, 'aav': 0.053},
    'W3': {'fzp': 31.67,   'lam': 11.56, 'aav': 0.020},
    'W4': {'fzp': 8.363,   'lam': 22.09, 'aav': 0.010},
}

# ─── HD 159254 observed photometry ───────────────────────────────────
MAGS = {
    'B': 9.51,
    'G': 7.92, 'BP': 8.66, 'RP': 7.05,
    'J': 5.83, 'H': 5.37, 'Ks': 5.15,
    'W1': 4.30, 'W2': 4.49, 'W3': 5.04, 'W4': 4.96,
}
MAG_ERRS = {
    'B': 0.03,
    'G': 0.003, 'BP': 0.003, 'RP': 0.004,
    'J': 0.02, 'H': 0.04, 'Ks': 0.02,
    'W1': 0.12, 'W2': 0.07, 'W3': 0.01, 'W4': 0.03,
}
BP_RP_OBS = 1.616
DIST_PC = 2527
DM = 5 * np.log10(DIST_PC / 10)  # distance modulus

# Intrinsic BP-RP for various Teff (approx from Pecaut & Mamajek 2013)
BPRP_INTRINSIC = {
    3500: 2.80, 3800: 2.35, 4000: 2.10, 4200: 1.90,
    4500: 1.70, 4800: 1.50, 5000: 1.40, 5200: 1.25,
    5500: 1.10, 5800: 0.95, 6000: 0.85, 6250: 0.75,
    6500: 0.65, 7000: 0.53, 7500: 0.39, 8000: 0.27,
    8500: 0.18, 9000: 0.10, 9500: 0.03, 10000: -0.02,
    11000: -0.10, 12000: -0.14, 15000: -0.25, 20000: -0.40,
    25000: -0.50,
}


def blackbody_flux(T, lam_um):
    """Planck function B_ν(T) in Jy-like units."""
    lam_cm = lam_um * 1e-4
    nu = c_cgs / lam_cm
    x = h_cgs * nu / (k_cgs * T)
    if x > 500:
        return 0.0
    return (2 * h_cgs * nu**3 / c_cgs**2) / (np.exp(x) - 1) / Jy


def fit_sed(av, teff, use_bands=None):
    """Fit single-star blackbody SED to dereddened photometry."""
    if use_bands is None:
        use_bands = list(MAGS.keys())
    bands, obs_flux, obs_err, model_flux = [], [], [], []

    for band in use_bands:
        mag = MAGS[band]
        f = FILTERS[band]
        a_band = f['aav'] * av
        mag_dered = mag - a_band
        flux = f['fzp'] * 10**(-0.4 * mag_dered)
        err_flux = flux * 0.4 * np.log(10) * MAG_ERRS[band]
        bands.append(band)
        obs_flux.append(flux)
        obs_err.append(max(err_flux, flux * 0.03))
        model_flux.append(blackbody_flux(teff, f['lam']))

    obs_flux = np.array(obs_flux)
    obs_err = np.array(obs_err)
    model_flux = np.array(model_flux)

    scale = np.sum(obs_flux * model_flux / obs_err**2) / \
            np.sum(model_flux**2 / obs_err**2)
    model_scaled = scale * model_flux
    residuals = (obs_flux - model_scaled) / obs_err
    chi2 = np.sum(residuals**2)
    ndof = len(bands) - 2
    chi2_red = chi2 / max(ndof, 1)

    return bands, obs_flux, obs_err, model_scaled, residuals, chi2_red


def estimate_extinction_from_bprp(teff):
    """Estimate A_V from BP-RP excess for a given Teff."""
    temps = sorted(BPRP_INTRINSIC.keys())
    if teff <= temps[0]:
        bp_rp_0 = BPRP_INTRINSIC[temps[0]]
    elif teff >= temps[-1]:
        bp_rp_0 = BPRP_INTRINSIC[temps[-1]]
    else:
        for i in range(len(temps) - 1):
            if temps[i] <= teff <= temps[i + 1]:
                frac = (teff - temps[i]) / (temps[i + 1] - temps[i])
                bp_rp_0 = (BPRP_INTRINSIC[temps[i]] * (1 - frac) +
                           BPRP_INTRINSIC[temps[i + 1]] * frac)
                break

    ebr = max(BP_RP_OBS - bp_rp_0, 0)
    ag = ebr * 1.89   # A_G / E(BP-RP) ratio
    av = ag / 0.789   # A_V = A_G / (A_G/A_V)
    return bp_rp_0, ebr, ag, av


def grid_search():
    """Find best (Teff, A_V) pair by chi² minimisation."""
    best_chi2 = 1e30
    best_params = {}
    results_grid = []

    # Wide grid: 3500–25000 K in steps of 250
    for teff in range(3500, 25001, 250):
        bp_rp_0, ebr, ag, av = estimate_extinction_from_bprp(teff)
        if av < 0:
            continue
        bands, obs, err, model, res, chi2r = fit_sed(av, teff)
        results_grid.append({
            'Teff': teff, 'A_V': round(av, 3),
            'E_BPRP': round(ebr, 3), 'chi2_red': round(chi2r, 2)
        })
        if chi2r < best_chi2:
            best_chi2 = chi2r
            best_params = {
                'Teff': teff, 'A_V': round(av, 3),
                'E_BPRP': round(ebr, 3),
                'bp_rp_intrinsic': round(bp_rp_0, 3),
                'chi2_red': round(chi2r, 2),
                'bands': bands,
                'obs_flux': obs.tolist(),
                'model_flux': model.tolist(),
                'residuals': res.tolist(),
            }

    return best_params, results_grid


def estimate_primary_mass(teff, mg):
    """Rough primary mass estimate from Teff and absolute magnitude."""
    # Using empirical calibrations for evolved stars
    if teff < 4500:
        # Cool giant: use luminosity
        bc = -0.8 + 0.0001 * (teff - 3500)
        mbol = mg + bc
        log_l = (4.74 - mbol) / 2.5
        l = 10**log_l
        # Giant mass-luminosity (approximate)
        if l < 100:
            return 1.5
        elif l < 1000:
            return 3.0
        elif l < 5000:
            return 6.0
        else:
            return 10.0
    elif teff < 7000:
        # F/G range
        if mg < -3:
            return 5.0
        elif mg < -1:
            return 2.5
        else:
            return 1.5
    elif teff < 15000:
        # A/B range
        if mg < -5:
            return 12.0
        elif mg < -3:
            return 7.0
        elif mg < -1:
            return 4.0
        else:
            return 2.5
    else:
        # Hot OB
        if mg < -6:
            return 20.0
        elif mg < -4:
            return 12.0
        else:
            return 8.0


def make_figure(best, grid, figpath):
    """Create SED figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    teff = best['Teff']
    av = best['A_V']
    bands = best['bands']
    lams = [FILTERS[b]['lam'] for b in bands]

    # Left: observed vs dereddened
    obs_raw = []
    obs_dered = []
    for band in bands:
        f = FILTERS[band]
        flux_raw = f['fzp'] * 10**(-0.4 * MAGS[band])
        flux_dered = f['fzp'] * 10**(-0.4 * (MAGS[band] - f['aav'] * av))
        obs_raw.append(flux_raw)
        obs_dered.append(flux_dered)

    ax1.scatter(lams, obs_raw, c='red', s=60, zorder=5,
                label='Observed', marker='o')
    ax1.scatter(lams, obs_dered, c='blue', s=60, zorder=5,
                label=f'Dereddened ($A_V={av:.2f}$)', marker='s')
    ax1.plot(lams, best['model_flux'], 'k-', alpha=0.7, lw=1.5,
             label=f'BB $T_{{eff}}={teff}$ K')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'Wavelength ($\mu$m)')
    ax1.set_ylabel('Flux density (Jy)')
    ax1.set_title(
        f'SED ($T_{{\\mathrm{{eff}}}}={teff}$ K, $A_V={av:.2f}$)')
    ax1.legend(fontsize=8)
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.set_xlim(0.3, 30)

    # Right: chi2 vs Teff grid
    grid_teff = [g['Teff'] for g in grid]
    grid_chi2 = [g['chi2_red'] for g in grid]
    ax2.plot(grid_teff, grid_chi2, 'b.-', ms=4, lw=1)
    ax2.axvline(teff, color='red', ls='--', alpha=0.7,
                label=f'Best fit: {teff} K')
    ax2.set_xlabel(r'$T_\mathrm{eff}$ (K)')
    ax2.set_ylabel(r'Reduced $\chi^2$')
    ax2.set_title(r'SED fit quality vs $T_\mathrm{eff}$')
    ax2.legend(fontsize=9)
    ax2.set_yscale('log')

    fig.suptitle(r'HD 159254 — SED Analysis', fontsize=13,
                 fontweight='bold')
    fig.tight_layout()
    os.makedirs(os.path.dirname(figpath), exist_ok=True)
    fig.savefig(figpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  -> {figpath}')


def main():
    print('=== SED + Extinction Analysis for HD 159254 ===\n')
    print(f'  Observed: G={MAGS["G"]}, BP-RP={BP_RP_OBS}')
    print(f'  Direction: l=0.59, b=+3.20 (toward Galactic Centre)')
    print(f'  Distance: {DIST_PC} pc, DM={DM:.2f}')

    # Grid search for best Teff, A_V
    best, grid = grid_search()

    teff = best['Teff']
    av = best['A_V']
    print(f'\n  Best-fit Teff = {teff} K')
    print(f'  Best-fit A_V  = {av:.2f} mag')
    print(f'  E(BP-RP)      = {best["E_BPRP"]:.3f} mag')
    print(f'  (BP-RP)_0     = {best["bp_rp_intrinsic"]:.3f}')
    print(f'  Chi2_red      = {best["chi2_red"]:.2f}')

    # Absolute magnitude
    ag = av * 0.789
    mg = MAGS['G'] - ag - DM
    print(f'\n  A_G = {ag:.2f} mag')
    print(f'  DM  = {DM:.2f} mag')
    print(f'  M_G = {mg:.2f} mag (corrected)')

    # Primary mass estimate
    m1_est = estimate_primary_mass(teff, mg)
    print(f'\n  Estimated M1 ~ {m1_est:.1f} Msun (from Teff + M_G)')

    # Grid summary (every 4th entry)
    print(f'\n  Teff grid results (selected):')
    print(f'  {"Teff":>6} {"A_V":>6} {"E(BP-RP)":>9} {"chi2_red":>9}')
    for i, g in enumerate(grid):
        if i % 4 == 0 or g['Teff'] == teff:
            flag = ' <-- best' if g['Teff'] == teff else ''
            print(f'  {g["Teff"]:6d} {g["A_V"]:6.2f} {g["E_BPRP"]:9.3f}'
                  f' {g["chi2_red"]:9.2f}{flag}')

    # Save results
    basedir = os.path.dirname(__file__)
    results = {
        'best_Teff_K': teff,
        'best_A_V': av,
        'E_BP_RP': best['E_BPRP'],
        'bp_rp_intrinsic': best['bp_rp_intrinsic'],
        'bp_rp_observed': BP_RP_OBS,
        'chi2_red': best['chi2_red'],
        'A_G': round(ag, 2),
        'M_G_corrected': round(mg, 2),
        'distance_modulus': round(DM, 2),
        'M1_estimate': m1_est,
        'grid': grid,
    }
    outpath = os.path.join(basedir, '..', 'results', 'sed_fit_results.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Saved: {outpath}')

    # Figure
    figpath = os.path.join(basedir, '..', 'paper', 'figures',
                           'fig_sed_extinction.pdf')
    make_figure(best, grid, figpath)

    print('\n=== SED analysis complete ===')


if __name__ == '__main__':
    main()
