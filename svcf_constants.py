"""
svcf_constants.py
SVCF Fundamental Constants and Domain Parameters

All constants derived from published repository papers.
rxiVerse:2602.0018  |  Zenodo: 10.5281/zenodo.18604376
"""

import numpy as np

# ============================================================
# SVCF FRAMEWORK CONSTANTS
# ============================================================

GAMMA    = 1.0 / 2857.0      # Substrate coupling constant (dimensionless)
RE_CRIT  = 2857.0             # Critical Reynolds number (= 1/GAMMA)
B        = 32.0 / 33.0        # Brinkman screening factor (Chirality Tax)
PSI      = np.sqrt(2) - 1     # Phase alignment factor (silver ratio conjugate)
K_TD     = 11100               # Topological degeneracy factor
K_MODE   = 9                   # Photon quantized drift mode (T^7 KK compact sector)
N_INTERNAL = 33                # Internal degrees of freedom (1 temporal + 7 compact + 25 active)

# ============================================================
# NUCLEAR MAGIC NUMBER CONSTANTS
# ============================================================

SHELL_DEGENERACY = [2, 6, 12, 8, 22, 32, 44]    # Shell occupation numbers (d_n)
MAGIC_NUMBERS    = [2, 8, 20, 28, 50, 82, 126]   # Nuclear magic numbers (cumsum of SHELL_DEGENERACY)

# ============================================================
# PHYSICAL CONSTANTS (CODATA 2018)
# ============================================================

HBAR       = 1.054571817e-34    # Reduced Planck constant (J·s)
C_LIGHT    = 2.99792458e8       # Speed of light in vacuum (m/s)
K_BOLTZ    = 1.380649e-23       # Boltzmann constant (J/K)
M_PROTON   = 1.67262192369e-27  # Proton mass (kg)
M_ELECTRON = 9.1093837015e-31   # Electron mass (kg)
ALPHA_FS   = 7.2973525693e-3    # Fine-structure constant (≈ 1/137, dimensionless)
EPSILON    = 8.854187817e-12    # Permittivity of free space (F/m)

# ============================================================
# SUBSTRATE VISCOSITY PARAMETERS
# ============================================================

# ETA derived from: tau_dec(300 K) = hbar^2 / (2 * Re_crit * eta * k_B * T) = 0.69 ys
ETA   = 6.810178e-28            # Substrate dynamic viscosity (Pa·s)
RHO_C = 9.47e-27                # Critical substrate density (kg/m^3)
NU    = ETA / RHO_C             # Substrate kinematic viscosity (m^2/s)
