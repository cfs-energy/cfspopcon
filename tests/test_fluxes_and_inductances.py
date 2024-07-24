import numpy as np
import xarray as xr
from scipy import constants

from cfspopcon.formulas.plasma_current.flux_consumption import (
    calc_external_flux,
    calc_external_inductance,
    calc_internal_flux,
    calc_internal_inductance_for_cylindrical,
    calc_internal_inductivity,
    calc_invmu_0_dLedR,
    calc_poloidal_field_flux,
    calc_resistive_flux,
    calc_vertical_field_mutual_inductance,
    calc_vertical_magnetic_field,
)
from cfspopcon.formulas.plasma_current.flux_consumption.inductance_analytical_functions import (
    calc_fa,
    calc_fa_Sum_Ne,
    calc_fa_Sums_Na,
    calc_fb,
    calc_fb_Sum_Nb,
    calc_fc,
    calc_fc_Sum_Nc,
    calc_fd,
    calc_fd_Sum_Nd,
    calc_fg,
    calc_fg_Sum_Ce,
    calc_fg_Sums_Na,
    calc_fh,
    calc_fh_Sum_Cb,
)
from cfspopcon.formulas.plasma_current.flux_consumption.inductances import (
    set_surface_inductance_coeffs,
)
from cfspopcon.named_options import SurfaceInductanceCoeffs, VertMagneticFieldEq
from cfspopcon.unit_handling import magnitude_in_units as umag
from cfspopcon.unit_handling import ureg


def test_calc_flux_internal():
    internal_inductance = calc_internal_inductance_for_cylindrical(1.85 * ureg.m, 0.8)
    internal_flux = calc_internal_flux(8.7e6 * ureg.A, internal_inductance)
    np.testing.assert_allclose(
        umag(internal_flux, ureg.weber), 8.090229405928559, rtol=1e-5, atol=0
    )


def test_calc_flux_external():
    coeffs = SurfaceInductanceCoeffs.Barr
    external_inductance = calc_external_inductance(
        0.3, 1.7, 0.3, 1.85 * ureg.m, 0.8, coeffs
    )
    external_flux = calc_external_flux(8.7e6 * ureg.A, external_inductance)
    np.testing.assert_allclose(
        umag(external_flux, ureg.weber), 24.21256335948457, rtol=1e-5, atol=0
    )


def test_calc_flux_PF():
    coeffs = SurfaceInductanceCoeffs.Barr
    vertical_field_mutual_inductance = calc_vertical_field_mutual_inductance(
        0.3, 1.7, coeffs
    )
    poloidal_field_flux = calc_poloidal_field_flux(
        vertical_field_mutual_inductance, 0.6 * ureg.T, 1.85 * ureg.m
    )
    np.testing.assert_allclose(
        umag(poloidal_field_flux, ureg.weber), 6.629771312025196, rtol=1e-5, atol=0
    )


def test_calc_flux_resistive():
    resistive_flux = calc_resistive_flux(8.7e6 * ureg.A, 1.85 * ureg.m, 0.45)
    np.testing.assert_allclose(
        umag(resistive_flux, ureg.weber), 9.10150808166963, rtol=1e-5, atol=0
    )


def test_calc_internal_inductivity():
    plasma_current = 8.7e6  # * ureg.A
    major_radius = 1.85  # * ureg.m
    magnetic_field_on_axis = 12.2  # * ureg.T
    minor_radius = 0.3 * 1.85  # * ureg.m

    cylindrical_safety_factor = (
        2
        * np.pi
        * minor_radius**2
        * magnetic_field_on_axis
        / (constants.mu_0 * major_radius * plasma_current)
    )

    internal_inductivity = calc_internal_inductivity(cylindrical_safety_factor, 1.01)
    np.testing.assert_allclose(
        internal_inductivity, 0.5814953402118008, rtol=1e-5, atol=0
    )


def test_calc_vertical_field_mutual_inductance():
    coeffs = SurfaceInductanceCoeffs.Barr
    vertical_field_mutual_inductance = calc_vertical_field_mutual_inductance(
        0.3, 1.7, coeffs
    )
    np.testing.assert_allclose(
        vertical_field_mutual_inductance, 1.027670685052496, rtol=1e-5, atol=0
    )


def test_calc_vertical_magnetic_field():
    coeffs = SurfaceInductanceCoeffs.Barr
    external_inductance = calc_external_inductance(
        0.3, 1.7, 0.3, 1.85 * ureg.m, 0.8, coeffs
    )
    vertical_magnetic_field_equation = VertMagneticFieldEq.Barr
    invmu_0_dLedR = calc_invmu_0_dLedR(
        0.3, 1.7, 0.3, 0.8, external_inductance, 1.85 * ureg.m, coeffs
    )
    vertical_magnetic_field = calc_vertical_magnetic_field(
        0.3,
        1.7,
        0.3,
        0.8,
        external_inductance,
        1.85 * ureg.m,
        8.7e6 * ureg.A,
        invmu_0_dLedR,
        vertical_magnetic_field_equation,
        coeffs,
    )
    np.testing.assert_allclose(
        umag(vertical_magnetic_field, ureg.tesla), 1.1286206938400316, rtol=1e-5, atol=0
    )


def test_calc_external_inductance():
    coeffs = SurfaceInductanceCoeffs.Barr
    external_inductance = calc_external_inductance(
        0.3, 1.7, 0.3, 1.85 * ureg.m, 0.8, coeffs
    )
    np.testing.assert_allclose(
        umag(external_inductance, ureg.henry), 2.78305325971087e-06, rtol=1e-5, atol=0
    )


def test_calc_invmu_0_dLedR():
    coeffs = SurfaceInductanceCoeffs.Barr
    external_inductance = calc_external_inductance(
        0.3, 1.7, 0.3, 1.85 * ureg.m, 0.8, coeffs
    )
    invmu_0_dLedR = calc_invmu_0_dLedR(
        0.3, 1.7, 0.3, 0.8, external_inductance, 1.85 * ureg.m, coeffs
    )
    np.testing.assert_allclose(
        umag(invmu_0_dLedR, ureg.dimensionless), 2.1999405545602646, rtol=1e-5, atol=0
    )


def test_calc_fa():
    coeffs = SurfaceInductanceCoeffs.Barr
    fa = calc_fa(0.3, 0.3, 0.8, coeffs=set_surface_inductance_coeffs(coeffs))
    np.testing.assert_allclose(fa, 1.4335756919903129, rtol=1e-5, atol=0)


def test_calc_fa_Sum_Ne():
    coeffs = SurfaceInductanceCoeffs.Barr
    fa_Sum_Ne = calc_fa_Sum_Ne(0.3, coeffs=set_surface_inductance_coeffs(coeffs))
    np.testing.assert_allclose(fa_Sum_Ne, 0.024095227744249464, rtol=1e-5, atol=0)


def test_calc_fa_Sums_Na():
    coeffs = SurfaceInductanceCoeffs.Barr
    np.testing.assert_allclose(
        calc_fa_Sums_Na(0.3, coeffs=set_surface_inductance_coeffs(coeffs))[0],
        1.4293250376924287,
        rtol=1e-5,
        atol=0,
    )
    np.testing.assert_allclose(
        calc_fa_Sums_Na(0.3, coeffs=set_surface_inductance_coeffs(coeffs))[1],
        4.559771647300995,
        rtol=1e-5,
        atol=0,
    )


def test_calc_fb():
    coeffs = SurfaceInductanceCoeffs.Barr
    np.testing.assert_allclose(
        calc_fb(0.3, coeffs=set_surface_inductance_coeffs(coeffs)),
        0.08132941228621908,
        rtol=1e-5,
        atol=0,
    )


def test_calc_fb_Sum_Nb():
    coeffs = SurfaceInductanceCoeffs.Barr
    np.testing.assert_allclose(
        calc_fb_Sum_Nb(0.3, coeffs=set_surface_inductance_coeffs(coeffs)),
        -0.003446225999999998,
        rtol=1e-5,
        atol=0,
    )


def test_calc_fc():
    coeffs = SurfaceInductanceCoeffs.Barr
    np.testing.assert_allclose(
        calc_fc(0.3, coeffs=set_surface_inductance_coeffs(coeffs)),
        0.970874542,
        rtol=1e-5,
        atol=0,
    )


def test_calc_fc_Sum_Nc():
    coeffs = SurfaceInductanceCoeffs.Barr
    fc_Sum_Nc = calc_fc_Sum_Nc(0.3, coeffs=set_surface_inductance_coeffs(coeffs))
    np.testing.assert_allclose(fc_Sum_Nc, -0.029125457999999996, rtol=1e-5, atol=0)


def test_calc_fd():
    coeffs = SurfaceInductanceCoeffs.Barr
    fd = calc_fd(0.3, coeffs=set_surface_inductance_coeffs(coeffs))
    np.testing.assert_allclose(fd, 0.000826722, rtol=1e-5, atol=0)


def test_calc_fd_Sum_Nd():
    coeffs = SurfaceInductanceCoeffs.Barr
    fd_Sum_Nd = calc_fd_Sum_Nd(0.3, coeffs=set_surface_inductance_coeffs(coeffs))
    np.testing.assert_allclose(fd_Sum_Nd, -0.08141999999999998, rtol=1e-5, atol=0)


def test_calc_fg():
    coeffs = SurfaceInductanceCoeffs.Barr
    fg = calc_fg(0.3, 0.3, 0.8, coeffs=set_surface_inductance_coeffs(coeffs))
    np.testing.assert_allclose(fg, -3.2891718339522056, rtol=1e-5, atol=0)


def test_calc_fg_Sum_Ne():
    coeffs = SurfaceInductanceCoeffs.Barr
    fg_Sum_Ce = calc_fg_Sum_Ce(0.3, coeffs=set_surface_inductance_coeffs(coeffs))
    np.testing.assert_allclose(fg_Sum_Ce, 0.15044119670967493, rtol=1e-5, atol=0)


def test_calc_fg_Sums_Na():
    coeffs = SurfaceInductanceCoeffs.Barr
    np.testing.assert_allclose(
        calc_fg_Sums_Na(0.3, coeffs=set_surface_inductance_coeffs(coeffs))[0],
        3.4517083961540482,
        rtol=1e-5,
        atol=0,
    )
    np.testing.assert_allclose(
        calc_fg_Sums_Na(0.3, coeffs=set_surface_inductance_coeffs(coeffs))[1],
        11.39453620447642,
        rtol=1e-5,
        atol=0,
    )


def test_calc_fh():
    coeffs = SurfaceInductanceCoeffs.Barr
    fh = calc_fh(0.3, 1.7, coeffs=set_surface_inductance_coeffs(coeffs))
    np.testing.assert_allclose(fh, -0.780145019816599, rtol=1e-5, atol=0)


def test_calc_fh_Sum_Cb():
    coeffs = SurfaceInductanceCoeffs.Barr
    fh_Sum_Cb = calc_fh_Sum_Cb(0.3, coeffs=set_surface_inductance_coeffs(coeffs))
    np.testing.assert_allclose(fh_Sum_Cb, -0.024597188999999988, rtol=1e-5, atol=0)


def test_functionals_against_hirshman():
    epsilon = np.arange(0.1, 1, 0.1)
    coeffs = SurfaceInductanceCoeffs.Hirshman

    fa_ref = np.array([2.975, 2.285, 1.848, 1.507, 1.217, 0.957, 0.716, 0.487, 0.272])
    fb_ref = np.array([0.228, 0.325, 0.403, 0.465, 0.512, 0.542, 0.553, 0.538, 0.508])
    fc_ref = np.array([1.008, 1.038, 1.093, 1.177, 1.301, 1.486, 1.769, 2.223, 2.864])
    fd_ref = np.array([0.022, 0.056, 0.087, 0.113, 0.134, 0.148, 0.155, 0.152, 0.133])

    for i, eps in enumerate(epsilon):
        fa = calc_fa(eps, 0, 0, coeffs=set_surface_inductance_coeffs(coeffs))
        fb = calc_fb(eps, coeffs=set_surface_inductance_coeffs(coeffs))
        fc = calc_fc(eps, coeffs=set_surface_inductance_coeffs(coeffs))
        fd = calc_fd(eps, coeffs=set_surface_inductance_coeffs(coeffs))

        assert np.allclose(fa, fa_ref[i], rtol=5e-2)
        assert np.allclose(fb, fb_ref[i], rtol=5e-2)
        assert np.allclose(fc, fc_ref[i], rtol=5e-2)
        assert np.allclose(fd, fd_ref[i], rtol=2.2e-1)


def test_inductances_against_hirshman():
    aspect_ratio_tests = np.array([20, 10, 5, 4, 3, 2])
    coeffs = SurfaceInductanceCoeffs.Hirshman
    kappa = 1.3

    vertical_mutual_inductance_ref = np.array([0.99, 0.96, 0.88, 0.83, 0.74, 0.52])
    normalized_external_inductance_ref = np.array([2.93, 2.22, 1.49, 1.26, 0.95, 0.53])

    for i, aspect_ratio in enumerate(aspect_ratio_tests):
        R0 = aspect_ratio * ureg.m

        vertical_mutual_inductance = calc_vertical_field_mutual_inductance(
            1 / aspect_ratio, kappa, coeffs
        )
        normalized_external_inductance = umag(
            (
                calc_external_inductance(1 / aspect_ratio, kappa, 0, R0, 0, coeffs)
                / (constants.mu_0 * aspect_ratio)
            ),
            ureg.henry,
        )

        assert np.isclose(
            vertical_mutual_inductance, vertical_mutual_inductance_ref[i], rtol=2e-2
        )
        assert np.isclose(
            normalized_external_inductance,
            normalized_external_inductance_ref[i],
            rtol=2e-2,
        )


def test_vertical_magnetic_field_against_Eq13():
    """Tests the vertical magnetic field equation derived by Mitarai & Takase (and used by Barr) against a
    typical formulation of the vertical field given in equation 13 in :cite:`mit&taka` at higher
    aspect ratios (where there should be agreement).
    """
    test_values = xr.DataArray(np.linspace(1.4, 4))

    inverse_aspect_ratio = 1.0 / test_values
    areal_elongation = 3
    beta_poloidal = 0.96
    major_radius = 2.8 * test_values * ureg.meter
    internal_inductivity = 0.5
    surface_inductance_coefficients = SurfaceInductanceCoeffs.Hirshman
    plasma_current = 48e6 * ureg.A

    Le_Mitarai = calc_external_inductance(
        inverse_aspect_ratio=inverse_aspect_ratio,
        areal_elongation=areal_elongation,
        beta_poloidal=beta_poloidal,
        major_radius=major_radius,
        internal_inductivity=internal_inductivity,
        surface_inductance_coefficients=surface_inductance_coefficients,
    )

    invmu_0_dLedR = calc_invmu_0_dLedR(
        inverse_aspect_ratio=inverse_aspect_ratio,
        areal_elongation=areal_elongation,
        beta_poloidal=beta_poloidal,
        internal_inductivity=internal_inductivity,
        external_inductance=Le_Mitarai,
        major_radius=major_radius,
        surface_inductance_coefficients=surface_inductance_coefficients,
    )

    Bv_Mitarai = calc_vertical_magnetic_field(
        inverse_aspect_ratio=inverse_aspect_ratio,
        areal_elongation=areal_elongation,
        beta_poloidal=beta_poloidal,
        internal_inductivity=internal_inductivity,
        external_inductance=Le_Mitarai,
        major_radius=major_radius,
        plasma_current=plasma_current,
        invmu_0_dLedR=invmu_0_dLedR,
        vertical_magnetic_field_equation=VertMagneticFieldEq.Barr,
        surface_inductance_coefficients=surface_inductance_coefficients,
    )

    Bv_Mit13 = calc_vertical_magnetic_field(
        inverse_aspect_ratio=inverse_aspect_ratio,
        areal_elongation=areal_elongation,
        beta_poloidal=beta_poloidal,
        internal_inductivity=internal_inductivity,
        external_inductance=Le_Mitarai,
        major_radius=major_radius,
        plasma_current=plasma_current,
        invmu_0_dLedR=invmu_0_dLedR,
        vertical_magnetic_field_equation=VertMagneticFieldEq.Mit_and_Taka_Eq13,
        surface_inductance_coefficients=surface_inductance_coefficients,
    )

    np.testing.assert_allclose(
        umag(Bv_Mitarai[17:-1], "tesla"), umag(Bv_Mit13[17:-1], "tesla"), rtol=2e-2
    )
