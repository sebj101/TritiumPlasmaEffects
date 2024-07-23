"""
Constants.py

Commonly used physical constants
"""

import numpy as np

# Tritium half life in years
HALFLIFETRITIUM = 12.32  # years

# One year in seconds
YEARSECONDS = 60.0 * 60.0 * 24.0 * 365.25  # seconds

# Tritium decay constant
DECAYCONSTTRITIUM = np.log(2) / HALFLIFETRITIUM / YEARSECONDS  # s^-1

# Rydberg energy in eV
RYDBERGEV = 13.605693122994  # eV

# Hydrogen ionization energy in eV
IONIZATIONENERGYH = 13.59844  # eV
