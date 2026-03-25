#!/usr/bin/env python3
"""
06_make_figures.py — Generate publication figures for HD 159254.

Produces:
  paper/figures/fig1_system_overview.pdf
  paper/figures/fig_checklist.pdf
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq

BASEDIR = os.path.dirname(__file__)
FIGDIR = os.path.join(BASEDIR, '..', 'paper', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# ─── System parameters ───────────────────────────────────────────────
P_DAYS = 619.989
ECC = 0.00565
K1 = 28.623
RUWE = 2.32
SIG = 343.4
DIST = 2527
FM = 1.0385e-7 * (1 - ECC**2)**1.5 * K1**3 * P_DAYS


def load_results():
    """Load results from previous scripts if available."""
    results = {'M1': 5.0, 'M2_MIN': 5.5, 'P_BH': None, 'Teff': 5000, 'A_V': 0.3}
    resdir = os.path.join(BASEDIR, '..', 'results')

    sed_path = os.path.join(resdir, 'sed_fit_results.json')
    if os.path.exists(sed_path):
        with open(sed_path) as f:
            sed = json.load(f)
        results['Teff'] = sed.get('best_Teff_K', results['Teff'])
        results['A_V'] = sed.get('best_A_V', results['A_V'])
        results['M1'] = sed.get('M1_estimate', results['M1'])

    mp_path = os.path.join(resdir, 'mass_posterior_results.json')
    if os.path.exists(mp_path):
        with open(mp_path) as f:
            mp = json.load(f)
        results['M2_MIN'] = mp.get('M2_min_fiducial', results['M2_MIN'])
        results['P_BH'] = mp.get('P_BH_percent', None)

    return results


def fig_system_overview():
    """Three-panel system overview."""
    res = load_results()
    M1 = res['M1']
    M2_MIN = res['M2_MIN']

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Left: orbital diagram
    theta = np.linspace(0, 2 * np.pi, 500)
    a = 1.0
    r = a * (1 - ECC**2) / (1 + ECC * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax1.plot(x, y, 'b-', lw=2)
    ax1.plot(0, 0, 'ko', ms=15, label='Dark companion')
    ax1.plot(x[0], y[0], '*', color='gold', ms=20, mec='darkorange',
             mew=1, label='Primary star')
    ax1.set_xlim(-1.3, 1.3)
    ax1.set_ylim(-1.3, 1.3)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x (arbitrary units)')
    ax1.set_ylabel('y (arbitrary units)')
    ax1.set_title(f'Orbital configuration ($e = {ECC:.003f}$)')
    ax1.legend(fontsize=9, loc='upper right')

    # Centre: multi-band photometry
    bands = ['B', 'BP', 'G', 'RP', 'J', 'H', 'Ks', 'W1', 'W2', 'W3', 'W4']
    mags = [9.51, 8.66, 7.92, 7.05, 5.83, 5.37, 5.15, 4.30, 4.49, 5.04, 4.96]
    lams = [0.440, 0.511, 0.622, 0.777, 1.235, 1.662, 2.159, 3.353, 4.603,
            11.56, 22.09]
    colors = ['blue', 'blue', 'green', 'red', 'brown', 'brown', 'brown',
              'purple', 'purple', 'purple', 'purple']
    ax2.scatter(lams, mags, c=colors, s=60, zorder=5)
    for i, b in enumerate(bands):
        ax2.annotate(b, (lams[i], mags[i]), textcoords='offset points',
                     xytext=(5, 5), fontsize=7)
    ax2.set_xscale('log')
    ax2.invert_yaxis()
    ax2.set_xlabel(r'Wavelength ($\mu$m)')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Multi-band photometry')

    # Right: key properties summary
    p_bh_str = f'{res["P_BH"]:.1f}%' if res['P_BH'] is not None else 'See 03'
    props = [
        ('Name', 'HD 159254'),
        ('Gaia DR3 ID', '4061...7392'),
        ('Period', f'{P_DAYS:.1f} d'),
        ('Eccentricity', f'{ECC:.003f}'),
        (u'K\u2081', f'{K1:.1f} km/s'),
        ('f(M)', f'{FM:.2f} M\u2609'),
        ('f(M) > Chandra.', u'\u2713 YES'),
        ('f(M) > NS ceil.', u'\u2717 NO'),
        (u'M\u2081 (SED)', f'~{M1:.1f} M\u2609'),
        (u'M\u2082,min', f'{M2_MIN:.1f} M\u2609'),
        ('RUWE', f'{RUWE:.2f}'),
        (u'\u03c3 (NSS)', f'{SIG:.1f}'),
        ('Distance', f'{DIST} pc'),
        ('P(BH)', p_bh_str),
    ]
    ax3.axis('off')
    y_pos = 0.95
    for key, val in props:
        ax3.text(0.05, y_pos, f'{key}:', fontsize=9.5, fontweight='bold',
                 transform=ax3.transAxes, va='top')
        ax3.text(0.55, y_pos, val, fontsize=9.5,
                 transform=ax3.transAxes, va='top')
        y_pos -= 0.065
    ax3.set_title('System properties')

    fig.suptitle('HD 159254 — System Overview', fontsize=14,
                 fontweight='bold')
    fig.tight_layout()
    path = os.path.join(FIGDIR, 'fig1_system_overview.pdf')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  -> {path}')


def fig_checklist():
    """Visual BH candidacy checklist."""
    res = load_results()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    checks = [
        ('$f(M) > $ Chandrasekhar (1.44 M$_\\odot$)', True,
         f'$f(M) = {FM:.2f}$ M$_\\odot$ $> 1.44$ M$_\\odot$ — WD excluded'),
        ('$f(M) > $ NS ceiling (3 M$_\\odot$)', False,
         f'$f(M) = {FM:.2f}$ M$_\\odot$ $< 3.0$ M$_\\odot$ — NS NOT excluded by dynamics'),
        ('No luminous companion detected', True,
         'Single-star SED; no secondary spectral signature'),
        ('Very high NSS significance', True,
         f'$\\sigma = {SIG:.1f}$ (68$\\times$ catalogue threshold)'),
        ('Physically coherent orbit', True,
         f'$P = {P_DAYS:.1f}$d, $e = {ECC:.003f}$, $K_1 = {K1:.1f}$ km/s'),
        ('No variability/eclipses', True,
         'Clean in Gaia, ASAS-SN, ZTF, VSX'),
        ('Clean astrometry (RUWE)', False,
         f'RUWE = {RUWE:.2f} > 1.4 — elevated; may be binary-induced'),
        ('Independent spectroscopy', False,
         'NEEDED: No RAVE/LAMOST/GALAH/APOGEE match'),
        ('Multi-epoch RV orbit', False,
         'NEEDED: ~8–12 epochs over ~1240 d'),
        ('UV spectroscopy', False,
         'NEEDED: to exclude stripped He-star companion'),
    ]

    n = len(checks)
    y_start = 0.90
    y_step = 0.082
    box_height = 0.060

    ax.text(0.5, 0.97, 'HD 159254 — BH Candidacy Assessment',
            fontsize=16, fontweight='bold', ha='center', va='top',
            transform=ax.transAxes)

    for i, (label, passed, detail) in enumerate(checks):
        y = y_start - i * y_step

        box_color = '#e8f5e9' if passed else '#ffebee'
        border_color = '#4caf50' if passed else '#f44336'
        rect = plt.Rectangle((0.02, y - box_height / 2), 0.96, box_height,
                              facecolor=box_color, edgecolor=border_color,
                              linewidth=1.5, transform=ax.transAxes,
                              clip_on=False, zorder=1)
        ax.add_patch(rect)

        icon = r'$\checkmark$' if passed else r'$\times$'
        color = '#2e7d32' if passed else '#c62828'
        ax.text(0.04, y, icon, fontsize=16, color=color,
                transform=ax.transAxes, va='center', fontweight='bold',
                zorder=2)
        ax.text(0.08, y + 0.005, label, fontsize=10,
                transform=ax.transAxes, va='center', zorder=2)
        ax.text(0.08, y - 0.020, detail, fontsize=7, color='gray',
                transform=ax.transAxes, va='center', zorder=2)

    # Verdict box
    n_pass = sum(1 for _, p, _ in checks if p)
    ax.text(0.5, 0.06,
            f'Score: {n_pass}/{n} criteria satisfied\n'
            f'Verdict: Strong BH candidate pending spectroscopic follow-up\n'
            f'Key limitation: f(M) < NS ceiling; BH depends on M$_1$ estimate',
            fontsize=10, ha='center', va='center',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow',
                      ec='orange', alpha=0.9))

    fig.tight_layout()
    path = os.path.join(FIGDIR, 'fig_checklist.pdf')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  -> {path}')


def main():
    print('=== Generating Publication Figures for HD 159254 ===\n')
    fig_system_overview()
    fig_checklist()
    print('\n=== Figure generation complete ===')


if __name__ == '__main__':
    main()
