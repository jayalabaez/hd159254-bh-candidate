# HD 159254 — Data Provenance Notes

## Target Identification
- **Name**: HD 159254 (SIMBAD)
- **Gaia DR3 source_id**: 4061400381769067392
- **SIMBAD object type**: SB* (spectroscopic binary)
- **B magnitude**: 9.51 (SIMBAD)
- **Coordinates**: RA = 263.6929°, Dec = -26.7397° (J2016.0)
- **Galactic**: l = 0.59°, b = +3.20°

## Data Sources
1. **Gaia DR3**: Astrometry, photometry (G, BP, RP), NSS SB1 orbit
2. **2MASS**: J = 5.83, H = 5.37, Ks = 5.15
3. **AllWISE**: W1 = 4.30, W2 = 4.49, W3 = 5.04, W4 = 4.96
4. **SIMBAD**: Identification, B magnitude, object type

## Spectroscopic Surveys
- **RAVE**: No match
- **LAMOST**: No match
- **GALAH**: No match
- **APOGEE**: No match
- No independent spectroscopic Teff, logg, or [Fe/H] available

## Key Caveats
- **Gaia GSP-Phot Teff = N/A**: No GSP-Phot parameters available for this source
- **No independent spectroscopy**: Primary mass must be estimated from SED + photometry alone
- **RUWE = 2.32**: Elevated above the canonical 1.4 threshold, indicating astrometric
  residuals beyond the single-star model. Could be related to the binary orbital motion
  or a partially resolved companion. Must be discussed as a caveat.
- **High extinction direction**: At l = 0.59°, b = +3.20° (toward Galactic Centre),
  significant interstellar extinction is expected. A_V is poorly constrained a priori.
- **2MASS/WISE saturation**: W1 and W2 have elevated errors (0.12, 0.07 mag)
  suggesting partial saturation for this G = 7.9 mag star.
- **Close neighbor**: 1 Gaia source within 3.3 arcsec. 2MASS (2–4" PSF) and
  WISE (6" beam) may be partially blended.
- **Near-zero eccentricity**: e = 0.006 ± 0.006 for a 620-day orbit is unusually
  circular, possibly indicating tidal circularisation from past mass transfer.
