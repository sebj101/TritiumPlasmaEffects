"""
Stone2002.py

Implementation of excitation cross-section for hydrogen using model from 
Stone, Kim & Desclaux (2002).
"""

import numpy as np
import scipy.constants as sc


class ExcitationXSec:
    """
    Class used for calculation of excitation cross-section for hydrogen.
    All cross sections are calculated from the 1s state.
    """

    def __init__(self, T):
        """
        Parametrised constructor for ExcitationXSec class.

        Parameters:
        ----------
        T : float
          Kinetic energy of primary in eV
        """
        self.__T = T

    def XSec2pData(self):
        """
        Excitation cross-section to 2p state for hydrogen

        Returns:
        --------
        np.ndarray, np.ndarray
          Energies and cross-sections for excitation to 2p state
        """
        energies = np.array([11., 12., 13., 14., 15., 16., 17., 18., 19.,
                             20., 21., 22., 23., 24., 26., 28., 30., 32.,
                             34., 36., 38., 40., 45., 50., 55., 60., 65.,
                             70., 75., 80., 85., 90., 95., 100., 110., 120.,
                             130., 140., 150., 160., 170., 180., 190., 200.,
                             250., 300., 350., 400., 450., 500., 600., 700.,
                             800., 900., 1000., 1500., 2000., 3000.])
        xsecs = np.array([0.15876, 0.24099, 0.31086, 0.35119, 0.39256,
                          0.42786, 0.45828, 0.48468, 0.50768, 0.52779,
                          0.54540, 0.56084, 0.57439, 0.58627, 0.60580,
                          0.62069, 0.63189, 0.64013, 0.64597, 0.64986,
                          0.65216, 0.65315, 0.65130, 0.64520, 0.63647,
                          0.62615, 0.61489, 0.60315, 0.59121, 0.57929,
                          0.56750, 0.55593, 0.54465, 0.53369, 0.51276,
                          0.49322, 0.47500, 0.45805, 0.44227, 0.42756,
                          0.41383, 0.40100, 0.38898, 0.37771, 0.33045,
                          0.29445, 0.26607, 0.24309, 0.22408, 0.20807,
                          0.18254, 0.16303, 0.14760, 0.13505, 0.12463,
                          0.09094, 0.07236, 0.05214])
        xsecs = xsecs * 1e-20
        return energies, xsecs

    def XSec3pData(self):
        """
        Excitation cross-section to 3p state for hydrogen

        Returns:
        --------
        np.ndarray, np.ndarray
          Energies and cross-sections for excitation to 2p state
        """
        energies = np.array([13., 14., 15., 16., 17., 18., 19.,
                             20., 21., 22., 23., 24., 26., 28., 30., 32.,
                             34., 36., 38., 40., 45., 50., 55., 60., 65.,
                             70., 75., 80., 85., 90., 95., 100., 110., 120.,
                             130., 140., 150., 160., 170., 180., 190., 200.,
                             250., 300., 350., 400., 450., 500., 600., 700.,
                             800., 900., 1000., 1500., 2000., 3000.])
        xsecs = np.array([0.03033, 0.04382, 0.05372, 0.06166, 0.06827, 0.07387,
                          0.07867, 0.08282, 0.08642, 0.08957, 0.09231, 0.09472,
                          0.09867, 0.10169, 0.10398, 0.10569, 0.10695, 0.10782,
                          0.10840, 0.10872, 0.10870, 0.10787, 0.10654, 0.10490,
                          0.10308, 0.10116, 0.09919, 0.09721, 0.09524, 0.09331,
                          0.09142, 0.08959, 0.08607, 0.08278, 0.07972, 0.07686,
                          0.07420, 0.07171, 0.06940, 0.06723, 0.06520, 0.06330,
                          0.05533, 0.04926, 0.04447, 0.04060, 0.03741, 0.03471,
                          0.03042, 0.02715, 0.02456, 0.02246, 0.02071, 0.01508,
                          0.01198, 0.00862])
        xsecs = xsecs * 1e-20
        return energies, xsecs

    def CalcXSec2p(self):
        """
        Excitation cross-section to 2p state for hydrogen

        Returns:
        --------
        float
          Excitation cross-section to 2p state
        """
        energies, xsecs = self.XSec2pData()
        if self.__T < 10.204:
            return 0.
        else:
            return np.interp(self.__T, energies, xsecs)

    def CalcXSec3p(self):
        """
        Excitation cross-section to 3p state for hydrogen

        Returns:
        --------
        float
          Excitation cross-section to 3p state
        """
        energies, xsecs = self.XSec3pData()
        if self.__T < 12.094:
            return 0.
        else:
            return np.interp(self.__T, energies, xsecs)
