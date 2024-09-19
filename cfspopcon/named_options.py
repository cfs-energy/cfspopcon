"""Enumerators to constrain options for functions."""

from enum import Enum, auto


class ProfileForm(Enum):
    """Methods to calculate nT profiles."""

    analytic = auto()
    prf = auto()


class RadiationMethod(Enum):
    """Methods to calculate radiation losses."""

    Inherent = "Bremsstrahlung and synchrotron radiation only"
    PostJensen = "Impurity radiation, using a coronal equilibrium model from Post & Jensen 1977"
    MavrinCoronal = "Impurity radiation, using a coronal equilibrium model from Mavrin 2018"
    MavrinNoncoronal = "Impurity radiation, using a non-coronal model from Mavrin 2017"
    Radas = "Impurity line and bremsstrahlung radiation, using coronal Lz curves from Radas"


class AtomicSpecies(Enum):
    """Enum of possible atomic species.

    The enum value represents the species atomic number (Z).
    """

    Hydrogen = 1
    Helium = 2
    Lithium = 3
    Beryllium = 4
    Carbon = 6
    Nitrogen = 7
    Oxygen = 8
    Neon = 10
    Argon = 18
    Krypton = 36
    Xenon = 54
    Tungsten = 74

    def __lt__(self, other: "AtomicSpecies") -> bool:
        """Implements '<' to allow sorting."""
        return self.value < other.value

    def __gt__(self, other: "AtomicSpecies") -> bool:
        """Implements '>' to allow sorting."""
        return self.value > other.value


class MomentumLossFunction(Enum):
    """Select which SOL momentum loss function to use."""

    KotovReiter = auto()
    Sang = auto()
    Jarvinen = auto()
    Moulton = auto()
    PerezH = auto()
    PerezL = auto()


class LambdaQScaling(Enum):
    """Options for heat flux decay length scaling."""

    Brunner = auto()
    EichRegression14 = auto()
    EichRegression15 = auto()


class ConfinementPowerScaling(Enum):
    """Options for which confinement threshold power scaling to use."""

    I_mode_AUG = auto()
    I_mode_HubbardNF17 = auto()
    I_mode_HubbardNF12 = auto()
    H_mode_Martin = auto()


class VertMagneticFieldEq(Enum):
    """Vertical magnetic field equation from various papers.

    NOTE: the choice of Barr vs. Mitarai also affects invmu_0_dLedR and the vertical_magnetic_field_mutual_inductance.
    """

    Mit_and_Taka_Eq13 = auto()
    Barr = auto()
    Jean = auto()
    MagneticFusionEnergyFormulary = auto()


class SurfaceInductanceCoeffs(Enum):
    """Coefficients to calculate external inductance components."""

    Hirshman = auto()
    Barr = auto()
