"""
Mott.py

Implementation of Mott cross-section for hydrogen
"""

import numpy as np
import scipy.constants as sc
from scipy.integrate import quad


class MottXSec:
    """
    Class used for calculation of Mott total and differential cross-sections.
    """

    def __init__(self, T):
        """
        Parametrised constructor for MottXSec class.

        Parameters:
        ----------
          T (float): Kinetic energy of primary in eV
        """
        self.__T = T

    def __RutherfordDCS(self, cosTheta):
        """
        Rutherford differential cross-section for hydrogen (Z=1)

        Parameters:
        ----------
          cosTheta (float): Cosine of scattering angle

        Returns:
        -------
          float: Rutherford differential cross-section in m^2
        """
        return (np.pi / 2.0) * sc.alpha**2 * (sc.hbar * sc.c / (self.__T * sc.e))**2 / (1 - cosTheta)**2

    def SinglyDifferentialXSec_theta(self, theta):
        """
        Singly differential cross-section (in theta) for hydrogen

        Parameters:
        ----------
          theta (float): Scattering angle in radians
        """
        cosTheta = np.cos(theta)
        ruthXSec = self.__RutherfordDCS(cosTheta)
        nuclMass = 3.01604928 * sc.m_u
        mottFactor = ((1 + cosTheta) / 2) / (1 + (1 - cosTheta)
                                             * self.__T * sc.e / (nuclMass * sc.c**2))
        return ruthXSec * mottFactor

    def TotalXSec(self):
        """
        Total Mott cross-section for hydrogen (Z=1)
        """
        def MottDCSTmp(cosTheta):
            ruthXSec = self.__RutherfordDCS(cosTheta)
            nuclMass = 3.01604928 * sc.m_u
            mottFactor = ((1 + cosTheta) / 2) / (1 + (1 - cosTheta)
                                                 * self.__T * sc.e / (nuclMass * sc.c**2))
            return ruthXSec * mottFactor

        return quad(lambda cosTheta: MottDCSTmp(cosTheta), -1, 1)[0]
