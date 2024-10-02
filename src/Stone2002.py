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
          Energies and cross-sections for excitation to 3p state
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

    def XSec4pData(self):
        """
        Excitation cross-section to 4p state for hydrogen

        Returns:
        --------
        np.ndarray, np.ndarray
          Energies and cross-sections for excitation to 4p state
        """
        energies = np.array([13., 14., 15., 16., 17., 18., 19.,
                             20., 21., 22., 23., 24., 26., 28., 30., 32.,
                             34., 36., 38., 40., 45., 50., 55., 60., 65.,
                             70., 75., 80., 85., 90., 95., 100., 110., 120.,
                             130., 140., 150., 160., 170., 180., 190., 200.,
                             250., 300., 350., 400., 450., 500., 600., 700.,
                             800., 900., 1000., 1500., 2000., 3000.])
        xsecs = np.array([0.00573, 0.01274, 0.01697, 0.02018, 0.02278, 0.02496,
                          0.02681, 0.02840, 0.02977, 0.03096, 0.03200, 0.03291,
                          0.03440, 0.03554, 0.03641, 0.03706, 0.03754, 0.03788,
                          0.03811, 0.03825, 0.03828, 0.03801, 0.03756, 0.03700,
                          0.03636, 0.03569, 0.03500, 0.03431, 0.03362, 0.03294,
                          0.03227, 0.03162, 0.03038, 0.02922, 0.02814, 0.02713,
                          0.02619, 0.02531, 0.02449, 0.02372, 0.02301, 0.02233,
                          0.01952, 0.01737, 0.01568, 0.01431, 0.01318, 0.01223,
                          0.01071, 0.00956, 0.00865, 0.00790, 0.00729, 0.00530,
                          0.00421, 0.00303])
        xsecs = xsecs * 1e-20
        return energies, xsecs

    def XSec5pData(self):
        """
        Excitation cross-section to 5p state for hydrogen

        Returns:
        --------
        np.ndarray, np.ndarray
          Energies and cross-sections for excitation to 5p state
        """
        energies = np.array([14., 15., 16., 17., 18., 19.,
                             20., 21., 22., 23., 24., 26., 28., 30., 32.,
                             34., 36., 38., 40., 45., 50., 55., 60., 65.,
                             70., 75., 80., 85., 90., 95., 100., 110., 120.,
                             130., 140., 150., 160., 170., 180., 190., 200.,
                             250., 300., 350., 400., 450., 500., 600., 700.,
                             800., 900., 1000., 1500., 2000., 3000.])
        xsecs = np.array([0.00528, 0.00752, 0.00916, 0.01046, 0.01154, 0.01245,
                          0.01323, 0.01391, 0.01449, 0.01500, 0.01544, 0.01617,
                          0.01673, 0.01715, 0.01747, 0.01771, 0.01787, 0.01799,
                          0.01806, 0.01808, 0.01796, 0.01775, 0.01749, 0.01719,
                          0.01688, 0.01655, 0.01622, 0.01590, 0.01557, 0.01526,
                          0.01495, 0.01437, 0.01382, 0.01331, 0.01283, 0.01238,
                          0.01197, 0.01158, 0.01122, 0.01088, 0.01056, 0.00923,
                          0.00821, 0.00741, 0.00676, 0.00623, 0.00578, 0.00506,
                          0.00452, 0.00408, 0.00373, 0.00344, 0.00250, 0.00199,
                          0.00143])
        xsecs = xsecs * 1e-20
        return energies, xsecs

    def XSec6pData(self):
        """
        Excitation cross-section to 6p state for hydrogen

        Returns:
        --------
        np.ndarray, np.ndarray
          Energies and cross-sections for excitation to 6p state
        """
        energies = np.array([14., 15., 16., 17., 18., 19.,
                             20., 21., 22., 23., 24., 26., 28., 30., 32.,
                             34., 36., 38., 40., 45., 50., 55., 60., 65.,
                             70., 75., 80., 85., 90., 95., 100., 110., 120.,
                             130., 140., 150., 160., 170., 180., 190., 200.,
                             250., 300., 350., 400., 450., 500., 600., 700.,
                             800., 900., 1000., 1500., 2000., 3000.])
        xsecs = np.array([0.00267, 0.00401, 0.00496, 0.00570, 0.00632, 0.00684,
                          0.00728, 0.00766, 0.00799, 0.00828, 0.00853, 0.00894,
                          0.00925, 0.00949, 0.00967, 0.00981, 0.00990, 0.00997,
                          0.01001, 0.01002, 0.00996, 0.00984, 0.00970, 0.00953,
                          0.00936, 0.00918, 0.00900, 0.00882, 0.00864, 0.00846,
                          0.00829, 0.00797, 0.00766, 0.00738, 0.00712, 0.00687,
                          0.00664, 0.00642, 0.00622, 0.00603, 0.00586, 0.00512,
                          0.00455, 0.00411, 0.00375, 0.00345, 0.00320, 0.00281,
                          0.00250, 0.00226, 0.00207, 0.00191, 0.00139, 0.00110,
                          0.000791])
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

    def CalcXSec4p(self):
        """
        Excitation cross-section to 4p state for hydrogen

        Returns:
        --------
        float
          Excitation cross-section to 4p state
        """
        energies, xsecs = self.XSec4pData()
        if self.__T < 12.755:
            return 0.
        else:
            return np.interp(self.__T, energies, xsecs)

    def CalcXSec5p(self):
        """
        Excitation cross-section to 5p state for hydrogen

        Returns:
        --------
        float
          Excitation cross-section to 5p state
        """
        energies, xsecs = self.XSec5pData()
        if self.__T < 13.061:
            return 0.
        else:
            return np.interp(self.__T, energies, xsecs)

    def CalcXSec6p(self):
        """
        Excitation cross-section to 6p state for hydrogen

        Returns:
        --------
        float
          Excitation cross-section to 6p state
        """
        energies, xsecs = self.XSec6pData()
        if self.__T < 13.228:
            return 0.
        else:
            return np.interp(self.__T, energies, xsecs)

    def TotalXSec(self):
        """
        Total excitation cross-section for hydrogen

        Returns:
        --------
        float
          Total excitation cross-section for hydrogen [m^2]
        """
        return (self.CalcXSec2p() + self.CalcXSec3p() + self.CalcXSec4p() +
                self.CalcXSec5p() + self.CalcXSec6p())
