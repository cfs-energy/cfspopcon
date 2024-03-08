"""Enumerators to constrain options for functions."""
from enum import Enum, auto


class Algorithms(Enum):
    """Select which top-level algorithm to run."""

    predictive_popcon = auto()
    two_point_model_fixed_fpow = auto()
    two_point_model_fixed_qpart = auto()
    two_point_model_fixed_tet = auto()
    calc_beta = auto()
    calc_core_radiated_power = auto()
    calc_fusion_gain = auto()
    calc_geometry = auto()
    calc_heat_exhaust = auto()
    calc_ohmic_power = auto()
    calc_peaked_profiles = auto()
    calc_plasma_current_from_q_star = auto()
    calc_q_star_from_plasma_current = auto()
    calc_power_balance_from_tau_e = auto()
    calc_zeff_and_dilution_from_impurities = auto()
    calc_confinement_transition_threshold_power = auto()
    calc_ratio_P_LH = auto()
    calc_f_rad_core = auto()
    calc_normalised_collisionality = auto()
    calc_rho_star = auto()
    calc_triple_product = auto()
    calc_greenwald_fraction = auto()
    calc_current_relaxation_time = auto()
    calc_peak_pressure = auto()
    calc_average_total_pressure = auto()
    calc_bootstrap_fraction = auto()
    calc_auxillary_power = auto()
    calc_average_ion_temp = auto()
    calc_fuel_average_mass_number = auto()
    calc_magnetic_field_on_axis = auto()
    calc_extrinsic_core_radiator = auto()
    require_P_rad_less_than_P_in = auto()
    calc_P_SOL = auto()
    use_LOC_tau_e_below_threshold = auto()
    calc_plasma_stored_energy = auto()
    calc_line_averaged_density = auto()


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


class Impurity(Enum):
    """Enum of possible impurity elements.

    The enum value represents the element's atomic number (Z).
    """

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
