"""
Rudd1991.py

Implementation of ionisation cross-sections for hydrogen derived from:
M. E. Rudd, Phys. Rev. A 44 (1991)
"""

import numpy as np
import scipy.constants as sc
import Constants as myconst


class RuddXSec:
    """
    Class used for calculation of cross-sections for hydrogen ionisation.
    """

    def __init__(self, T):
        """
        Parametrised constructor for RuddXSec class.

        Parameters:
        ----------
          T (float): Kinetic energy of primary in eV
        """
        self.__T = T
        self.__t = self.__T / myconst.IONIZATIONENERGYH

    def __S():
        return 4.0 * np.pi * sc.physical_constants['Bohr radius'][0]**2 * (myconst.RYDBERGEV / myconst.IONIZATIONENERGYH)**2

    def __F(self):
        A1 = 0.74
        A2 = 0.87
        A3 = -0.6
        return (A1 * np.log(self.__t) + A2 + A3 / self.__t) / self.__t

    def __g_1(self):
        n = 2.4
        return (1.0 - self.__t**(1 - n)) / (n - 1.0) - (2.0 / (self.__t + 1))**(n / 2.0) * (1 - self.__t**(1 - n / 2)) / (n - 2)

    def TotalXSec(self):
        """
        Total ionisation cross-section from Rudd

        Returns:
        -------
          float: Total ionisation cross-section in m^2
        """
        return self.__S() * self.__F() * self.__g_1()

    def __f_1(self, W):
        w = W / myconst.IONIZATIONENERGYH
        n = 2.4
        return 1.0 / (w + 1)**n + 1.0 / (self.__t - w)**n - 1.0 / ((w + 1) * (self.__t - w))**(n / 2.0)

    def __G2(self, W):
        w = W / myconst.IONIZATIONENERGYH
        return np.sqrt((w + 1.0) / self.__t)

    def __G3(self, W):
        w = W / myconst.IONIZATIONENERGYH
        beta = 0.6
        return beta * np.sqrt((1.0 - self.__G2(W)**2) / w)

    def __G4(self, W):
        w = W / myconst.IONIZATIONENERGYH
        gamma = 10.0
        return gamma * (1.0 - w / self.__t)**3 / (self.__t * (w + 1.0))

    def __gBE(self, W):
        return 2.0 * np.pi * self.__G3(W) * (np.arctan((1.0 - self.__G2(W)) / self.__G3(W)) + np.arctan((1.0 + self.__G2(W)) / self.__G3(W)))

    def __G1(self, W):
        return (self.__S() * self.__F() * self.__f_1(W) / myconst.IONIZATIONENERGYH) / (self.__gBE(W) + self.__G4(W) * 2.9)

    def SinglyDifferentialXSec_W(self, W):
        """
        Singly differential cross-section in W (outgoing electron energy)

        Paramters:
        ---------
          W (float): Outgoing electron energy in eV
        """
        return self.__G1(W) * (self.__gBE(W) + self.__G4(W) * 2.9)

    def __fBE(self, W, theta):
        return 1.0 / (1.0 + ((np.cos(theta) - self.__G2(W)) / self.__G3(W))**2)

    def __G4fb(self, W, theta):
        w = W / myconst.IONIZATIONENERGYH
        G5 = 0.33
        return self.__G4(W) / (1.0 + ((np.cos(theta) + 1) / G5)**2)

    def DoublyDifferentialXSec(self, W, theta):
        """
        Doubly differential cross-section in W and theta

        Parameters:
        ----------
          W (float): Outgoing electron energy in eV
          theta (float): Scattering angle in radians
        """
        return self.__G1(W) * (self.__fBE(W, theta) + self.__G4fb(W, theta))
