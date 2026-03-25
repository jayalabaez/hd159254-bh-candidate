#!/usr/bin/env python3
"""
01_compile_data.py — Compile all public archival data for HD 159254.

All data were retrieved from Gaia DR3, 2MASS, AllWISE, and SIMBAD.
This script writes the compiled dataset to data/ and results/ for
downstream analysis.

Outputs:
  data/hd159254_input_summary.csv
  data/photometry_compiled.csv
  results/compiled_data.json
"""

import json, os, csv, math

BASEDIR = os.path.dirname(__file__)
DATADIR = os.path.join(BASEDIR, '..', 'data')
RESDIR = os.path.join(BASEDIR, '..', 'results')
os.makedirs(DATADIR, exist_ok=True)
os.makedirs(RESDIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
#  All values retrieved from public archives (Gaia, 2MASS, WISE, SIMBAD)
# ═══════════════════════════════════════════════════════════════════════
data = {
    'target': {
        'name': 'HD 159254',
        'gaia_source_id': 4061400381769067392,
        'simbad_type': 'SB*',
        'B_mag': 9.51,
    },
    'gaia_astrometry': {
        'ra_deg': 263.6929,
        'dec_deg': -26.7397,
        'l_deg': 0.59,
        'b_deg': 3.20,
        'parallax_mas': 0.3958,
        'parallax_error_mas': 0.091,
        'distance_pc': 2527,
        'pmra_mas_yr': None,
        'pmdec_mas_yr': None,
        'ruwe': 2.32,
    },
    'gaia_photometry': {
        'G': 7.92,
        'BP': 8.66,
        'RP': 7.05,
        'BP_RP': 1.616,
        'G_err': 0.003,
        'BP_err': 0.003,
        'RP_err': 0.004,
    },
    'gaia_gspphot': {
        'teff_K': None,
        'logg': None,
        'note': 'No GSP-Phot parameters available for this source',
    },
    'nss_orbital': {
        'solution_type': 'SB1',
        'period_d': 619.989,
        'period_err_d': 1.449,
        'eccentricity': 0.00565,
        'eccentricity_err': 0.00615,
        'K1_km_s': 28.623,
        'K1_err_km_s': 0.083,
        'significance': 343.4,
        'goodness_of_fit': 1.62,
        'rv_template_teff_K': None,
    },
    'gaia_rv': {
        'rv_km_s': -13.02,
        'rv_err_km_s': 9.58,
    },
    'twomass': {
        'J': 5.83, 'J_err': 0.02,
        'H': 5.37, 'H_err': 0.04,
        'Ks': 5.15, 'Ks_err': 0.02,
        'note': 'Close neighbor at 3.3 arcsec may cause partial blending',
    },
    'allwise': {
        'W1': 4.30, 'W1_err': 0.12,
        'W2': 4.49, 'W2_err': 0.07,
        'W3': 5.04, 'W3_err': 0.01,
        'W4': 4.96, 'W4_err': 0.03,
        'note': 'W1/W2 errors elevated (partial saturation at G=7.9)',
    },
    'close_neighbor': {
        'n_within_5arcsec': 1,
        'closest_arcsec': 3.33,
        'note': 'Single Gaia source at 3.33 arcsec; too wide for 620d orbit',
    },
    'spectroscopic_surveys': {
        'RAVE': 'No match',
        'LAMOST': 'No match',
        'GALAH': 'No match',
        'APOGEE': 'No match',
        'note': 'No independent spectroscopy available',
    },
    'xray': {
        'ROSAT': 'Not queried',
        'XMM': 'Not queried',
        'note': 'X-ray constraints not yet obtained',
    },
    'variability': {
        'gaia_var_flag': False,
        'asassn_match': False,
        'ztf_match': False,
        'vsx_match': False,
        'eb_flag': False,
        'note': 'No variability detected in any survey',
    },
}


def compute_derived():
    """Compute derived quantities from the compiled data."""
    P = data['nss_orbital']['period_d']
    e = data['nss_orbital']['eccentricity']
    K1 = data['nss_orbital']['K1_km_s']

    # Mass function: f(M) = 1.0385e-7 * (1-e^2)^1.5 * K1^3 * P  [Msun]
    fM = 1.0385e-7 * (1 - e**2)**1.5 * K1**3 * P

    # Distance modulus
    d_pc = data['gaia_astrometry']['distance_pc']
    DM = 5 * math.log10(d_pc / 10)

    derived = {
        'mass_function_Msun': round(fM, 4),
        'distance_modulus': round(DM, 2),
        'note_fM': (
            f'f(M) = {fM:.4f} Msun. Exceeds Chandrasekhar limit '
            f'(1.44 Msun) but falls below the conservative NS ceiling '
            f'(3.0 Msun). The BH case therefore depends on the '
            f'primary mass M1 estimate.'
        ),
    }
    data['derived'] = derived
    return derived


def write_summary_csv():
    """Write summary CSV."""
    rows = [
        ('Name', 'HD 159254', '', 'SIMBAD'),
        ('Gaia source_id', '4061400381769067392', '', 'Gaia DR3'),
        ('RA (J2016.0)', '263.6929 deg', '', 'Gaia DR3'),
        ('Dec (J2016.0)', '-26.7397 deg', '', 'Gaia DR3'),
        ('l', '0.59 deg', '', 'Gaia DR3'),
        ('b', '+3.20 deg', '', 'Gaia DR3'),
        ('Parallax', '0.396 mas', '0.091', 'Gaia DR3'),
        ('Distance', '2527 pc', '', '1/parallax'),
        ('G', '7.92 mag', '0.003', 'Gaia DR3'),
        ('BP', '8.66 mag', '0.003', 'Gaia DR3'),
        ('RP', '7.05 mag', '0.004', 'Gaia DR3'),
        ('BP-RP', '1.616 mag', '', 'Gaia DR3'),
        ('B', '9.51 mag', '', 'SIMBAD'),
        ('RUWE', '2.32', '', 'Gaia DR3'),
        ('Period', '619.989 d', '1.449', 'Gaia NSS'),
        ('Eccentricity', '0.006', '0.006', 'Gaia NSS'),
        ('K1', '28.623 km/s', '0.083', 'Gaia NSS'),
        ('Significance', '343.4', '', 'Gaia NSS'),
        ('GoF', '1.62', '', 'Gaia NSS'),
        ('RV', '-13.02 km/s', '9.58', 'Gaia DR3'),
        ('J', '5.83 mag', '0.02', '2MASS'),
        ('H', '5.37 mag', '0.04', '2MASS'),
        ('Ks', '5.15 mag', '0.02', '2MASS'),
        ('W1', '4.30 mag', '0.12', 'AllWISE'),
        ('W2', '4.49 mag', '0.07', 'AllWISE'),
        ('W3', '5.04 mag', '0.01', 'AllWISE'),
        ('W4', '4.96 mag', '0.03', 'AllWISE'),
    ]
    path = os.path.join(DATADIR, 'hd159254_input_summary.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Property', 'Value', 'Uncertainty', 'Source'])
        w.writerows(rows)
    print(f'  Wrote {path}')


def write_photometry_csv():
    """Write photometry compiled CSV for SED fitting."""
    rows = [
        (0.440, 9.51, 0.03, 'B', 'SIMBAD'),
        (0.511, 8.66, 0.003, 'BP', 'Gaia'),
        (0.622, 7.92, 0.003, 'G', 'Gaia'),
        (0.777, 7.05, 0.004, 'RP', 'Gaia'),
        (1.235, 5.83, 0.02, 'J', '2MASS'),
        (1.662, 5.37, 0.04, 'H', '2MASS'),
        (2.159, 5.15, 0.02, 'Ks', '2MASS'),
        (3.353, 4.30, 0.12, 'W1', 'WISE'),
        (4.603, 4.49, 0.07, 'W2', 'WISE'),
        (11.56, 5.04, 0.01, 'W3', 'WISE'),
        (22.09, 4.96, 0.03, 'W4', 'WISE'),
    ]
    path = os.path.join(DATADIR, 'photometry_compiled.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['wavelength_um', 'magnitude', 'mag_err', 'filter', 'source'])
        w.writerows(rows)
    print(f'  Wrote {path}')


def main():
    print('=== Compiling data for HD 159254 ===\n')
    print(f'  Gaia DR3 {data["target"]["gaia_source_id"]}')
    print(f'  SIMBAD type: {data["target"]["simbad_type"]}')
    print(f'  B = {data["target"]["B_mag"]}')
    print(f'  G = {data["gaia_photometry"]["G"]}')
    print(f'  BP-RP = {data["gaia_photometry"]["BP_RP"]}')
    print(f'  d = {data["gaia_astrometry"]["distance_pc"]} pc')
    print(f'  RUWE = {data["gaia_astrometry"]["ruwe"]}')
    print(f'  P = {data["nss_orbital"]["period_d"]} d')
    print(f'  K1 = {data["nss_orbital"]["K1_km_s"]} km/s')
    print(f'  e = {data["nss_orbital"]["eccentricity"]}')
    print(f'  sigma = {data["nss_orbital"]["significance"]}')

    derived = compute_derived()
    fM = derived['mass_function_Msun']
    print(f'\n  f(M) = {fM:.4f} Msun')
    print(f'  f(M) > 1.44 Msun (Chandrasekhar):  {fM > 1.44}')
    print(f'  f(M) > 3.0 Msun (NS ceiling):      {fM > 3.0}')
    print(f'  DM = {derived["distance_modulus"]:.2f}')

    # Save JSON
    outpath = os.path.join(RESDIR, 'compiled_data.json')
    with open(outpath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f'\n  Saved: {outpath}')

    # Save CSVs
    write_summary_csv()
    write_photometry_csv()

    print('\n=== Data compilation complete ===')


if __name__ == '__main__':
    main()
