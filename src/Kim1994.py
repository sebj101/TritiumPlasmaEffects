"""
    Kim1994.py

    Class containing cross-sections from Kim & Rudd (1994) for electron-impact 
    ionization of atoms and molecules
"""

import numpy as np
import scipy.constants as sc


class Kim1994:
    def __init__(self, T: float, species: str):
        """
        Constructor for the Kim1994 class.

        Parameters
        ----------
        T : float
            Incident electron KE in eV
        species : str
            Species for which the cross-sections are calculated
        """
        self._T = T
        self._species = species

    def DiffOscillatorStrength(self, y: float) -> float:
        """
        Calculate the differential oscillator strength for electron-impact 
        ionization of atoms and molecules.

        Parameters
        ----------
        y : float
            Binding energy divided by the transfer energy

        Returns
        -------
        float
            Differential oscillator strength
        """
        if self._species == "H":
            a = 0.0
            b = -2.2473e-2
            c = 1.1775
            d = -4.6264e-1
            e = 8.9064e-2
            f = 0.0
        elif self._species == "He":
            a = 0.0
            b = 0.0
            c = 1.2178e1
            d = -2.9585e1
            e = 3.1251e1
            f = -1.2175e1
        elif self._species == "H2":
            a = 0.0
            b = 0.0
            c = 1.1262
            d = 6.3982
            e = -7.8055
            f = 2.1440
        else:
            raise ValueError("Species not supported")

        return a * y + b * y**2 + c * y**3 + d * y**4 + e * y**5 + f * y**6

    def DiffOscillatorStrength_w(self, w: float) -> float:
        """
        Calculate the differential oscillator strength for electron-impact 
        ionization of atoms and molecules.

        Parameters
        ----------
        w : float
            Outgoing electron energy divided by the binding energy

        Returns
        -------
        float
            Differential oscillator strength
        """
        if self._species == "H":
            a = 0.0
            b = -2.2473e-2
            c = 1.1775
            d = -4.6264e-1
            e = 8.9064e-2
            f = 0.0
        elif self._species == "He":
            a = 0.0
            b = 0.0
            c = 1.2178e1
            d = -2.9585e1
            e = 3.1251e1
            f = -1.2175e1
        elif self._species == "H2":
            a = 0.0
            b = 0.0
            c = 1.1262
            d = 6.3982
            e = -7.8055
            f = 2.1440
        else:
            raise ValueError("Species not supported")

        return a / (w + 1) + b / (w + 1)**2 + c / (w + 1)**3 + d / (w + 1)**4 + e / (w + 1)**5 + f / (w + 1)**6

    def _N(self) -> float:
        if self._species == "H":
            return 1.0
        elif self._species == "He":
            return 2.0
        elif self._species == "H2":
            return 2.0
        else:
            raise ValueError("Species not supported")

    def _B(self) -> float:
        """
        Calculate the binding energy of the target atom or molecule.

        Returns
        -------
        float
            Binding energy
        """
        if self._species == "H":
            return 1.36057e1
        elif self._species == "He":
            return 2.459e1
        elif self._species == "H2":
            return 1.543e1
        else:
            raise ValueError("Species not supported")

    def _D(self) -> float:
        t = self._T / self._B()
        if self._species == "H":
            a = 0.0
            b = -2.2473e-2
            c = 1.1775
            d = -4.6264e-1
            e = 8.9064e-2
            f = 0.0
        elif self._species == "He":
            a = 0.0
            b = 0.0
            c = 1.2178e1
            d = -2.9585e1
            e = 3.1251e1
            f = -1.2175e1
        elif self._species == "H2":
            a = 0.0
            b = 0.0
            c = 1.1262
            d = 6.3982
            e = -7.8055
            f = 2.1440
        else:
            raise ValueError("Species not supported")

        tTerm = (t + 1.0) / 2.0
        bTerm = (b / 2) * (1 - tTerm**-2)
        cTerm = (c / 3) * (1 - tTerm**-3)
        dTerm = (d / 4) * (1 - tTerm**-4)
        eTerm = (e / 5) * (1 - tTerm**-5)
        fTerm = (f / 6) * (1 - tTerm**-6)
        return (bTerm + cTerm + dTerm + eTerm + fTerm) / self._N()

    def _Ni(self) -> float:
        if self._species == "H":
            return 0.4343
        elif self._species == "He":
            return 1.605
        elif self._species == "H2":
            return 1.173

    def _U(self) -> float:
        """
        Calculate the average kinetic energy of the bound electrons

        Returns
        -------
        float
            Transfer energy
        """
        if self._species == "H":
            return 1.36057e1
        elif self._species == "He":
            return 3.951e1
        elif self._species == "H2":
            return 2.568e1
        else:
            raise ValueError("Species not supported")

    def _S(self) -> float:
        """
        Calculate the normalization factor

        Returns
        -------
        float
        """
        a0 = sc.physical_constants["Bohr radius"][0]
        bohrXSec = 4.0 * np.pi * a0**2
        return bohrXSec * self._N() * (sc.physical_constants["Rydberg constant times hc in eV"][0] / self._B())**2

    def SingleDiffXSec_W(self, W: float) -> float:
        """
        Calculate the SDCS for electron-impact ionization of atoms and 
        molecules.

        Parameters
        ----------
        W : float
            Outgoing electron energy in eV

        Returns
        -------
        float
            SDCS in m^2/eV
        """
        t = self._T / self._B()
        u = self._U() / self._B()
        w = W / self._B()
        prefac = self._S() / (self._B() * (t + u + 1))
        term1 = (self._Ni() / self._N() - 2) / \
            (t + 1) * (1 / (w + 1) + 1 / (t - w))
        term2 = (2 - self._Ni() / self._N()) * \
            (1 / (w + 1)**2 + 1 / (t - w)**2)
        term3 = np.log(t) / (self._N() * (w + 1)) * \
            self.DiffOscillatorStrength_w(w)
        return prefac * (term1 + term2 + term3)

    def TotalXSec(self) -> float:
        """
        Calculate the total cross-section for electron-impact ionization of 
        atoms and molecules.

        Returns
        -------
        float
            Total cross-section
        """
        t = self._T / self._B()
        u = self._U() / self._B()
        prefactor = self._S() / (t + u + 1)
        return prefactor * (self._D() * np.log(t) + (2 - self._Ni() / self._N()) * ((t - 1) / t - np.log(t) / (t + 1)))
