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


class ReactionType(Enum):
    """Supported Fusion Fuel Reaction Types."""

    DT = "Deuterium-Tritium"
    DD = "Deuterium-Deuterium"
    DHe3 = "Deuterium-Helium3"
    pB11 = "Proton-Boron11"


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


class ConfinementScaling(Enum):
    r"""Enum of implemented \tau_{E} scalings."""
    ITER98y2 = auto()
    ITER89P = auto()
    ITER89P_ka = auto()
    ITERL96Pth = auto()
    ITER97L = auto()
    IModey2 = auto()
    ITPA20_STD5 = auto()
    ITPA20_IL = auto()
    ITPA20_IL_HighZ = auto()
    ITPA_2018_STD5_OLS = auto()
    ITPA_2018_STD5_WLS = auto()
    ITPA_2018_STD5_GLS = auto()
    ITPA_2018_STD5_SEL1_OLS = auto()
    ITPA_2018_STD5_SEL1_WLS = auto()
    ITPA_2018_STD5_SEL1_GLS = auto()
    LOC = auto()
    H_DS03 = auto()


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
