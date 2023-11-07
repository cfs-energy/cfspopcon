import numpy as np
import xarray as xr
from scipy import constants
from cfspopcon.algorithms import fluxes, inductances

from cfspopcon.unit_handling import ureg
from cfspopcon.named_options import SurfaceInductanceCoeffs, InternalInductanceGeometry, VertMagneticFieldEq
from cfspopcon.formulas.fluxes import calc_flux_internal, calc_flux_external, calc_flux_PF, calc_flux_res
from cfspopcon.formulas.inductances import (
    calc_internal_inductance,
    calc_external_inductance,
    calc_internal_inductivity,
    calc_invmu_0_dLedR,
    calc_vertical_field_mutual_inductance,
    calc_vertical_magnetic_field,
    calc_fa_Sum_Ne,
    calc_fa,
    calc_fa_Sums_Na,
    calc_fb,
    calc_fb_Sum_Nb,
    calc_fc_Sum_Nc,
    calc_fc,
    calc_fd,
    calc_fd_Sum_Nd,
    calc_fg,
    calc_fg_Sum_Ce,
    calc_fg_Sums_Na,
    calc_fh,
    calc_fh_Sum_Cb,
)


def test_calc_inductances():
    inductances.run_calc_inductances(
        major_radius=1.85 * ureg.m,
        plasma_volume=19 * ureg.m**3,
        poloidal_circumference=4 * ureg.m,
        custom_internal_inductivity=False,
        internal_inductance_geometry=InternalInductanceGeometry.NonCylindrical,
        plasma_current=8.7 * ureg.MA,
        magnetic_field_on_axis=12.16 * ureg.T,
        minor_radius=0.56 * ureg.m,
        q_0=1,
        inverse_aspect_ratio=0.56 / 1.85,
        areal_elongation=1.7,
        beta_poloidal=0.3,
        vertical_magnetic_field_eq=VertMagneticFieldEq.Barr,
        surface_inductance_coefficients=SurfaceInductanceCoeffs.Barr,
    )


def test_calc_fluxes():
    fluxes.run_calc_fluxes(
        8.7 * ureg.MA,
        1.85 * ureg.m,
        internal_inductance=1e-6 * ureg.henry,
        external_inductance=2e-6 * ureg.henry,
        ejima_coefficient=0.4,
        vertical_field_mutual_inductance=4,
        vertical_magnetic_field=0.6 * ureg.tesla,
        loop_voltage=0.3 * ureg.volt,
        total_flux_available_from_CS=35 * ureg.weber,
    )


def test_calc_flux_internal():
    shape = InternalInductanceGeometry.Cylindrical
    internal_inductance = calc_internal_inductance(1.85 * ureg.m, 0.8, 0 * ureg.m**3, 0 * ureg.m, shape)
    internal_flux = calc_flux_internal(8.7e6 * ureg.A, internal_inductance).item()
    np.testing.assert_allclose(internal_flux, 8.090229405928559 * ureg.weber, rtol=5e-2, atol=0)


def test_calc_flux_external():
    coeffs = SurfaceInductanceCoeffs.Barr
    external_inductance = calc_external_inductance(
        0.3 * ureg.dimensionless, 1.7 * ureg.dimensionless, 0.3 * ureg.dimensionless, 1.85 * ureg.m, 0.8 * ureg.dimensionless, coeffs
    )
    external_flux = calc_flux_external(8.7e6 * ureg.A, external_inductance).item()
    np.testing.assert_allclose(external_flux, 24.21256335948457 * ureg.weber, rtol=5e-2, atol=0)


def test_calc_flux_PF():
    coeffs = SurfaceInductanceCoeffs.Barr
    vertical_field_mutual_inductance = calc_vertical_field_mutual_inductance(0.3, 1.7, coeffs)
    PF_flux = calc_flux_PF(vertical_field_mutual_inductance, 0.6 * ureg.T, 1.85 * ureg.m).item()
    np.testing.assert_allclose(PF_flux, 6.629771312025196 * ureg.weber, rtol=5e-2, atol=0)


def test_calc_flux_resistive():
    resistive_flux = calc_flux_res(8.7e6 * ureg.A, 1.85 * ureg.m, 0.45).item()
    assert resistive_flux == 9.10150808166963 * ureg.weber


def test_calc_internal_inductivity():
    internal_inductivity = calc_internal_inductivity(8.7e6 * ureg.A, 1.85 * ureg.m, 12.2 * ureg.T, 1.85 * ureg.m * 0.3, 1.01).item()
    np.testing.assert_allclose(internal_inductivity, 0.5814953402118008, rtol=5e-2, atol=0)


# def test_calc_internal_inductance():
#     calc_internal_inductance(1.85 * ureg.m,
#                             internal_inductivity: float,
#                             plasma_volume: float = 0 * ureg.meter ** 3,
#                             poloidal_circumference: float = 0 * ureg.meter,
#                             internal_inductance_geometry: InternalInductanceGeometry)


def test_calc_vertical_field_mutual_inductance():
    coeffs = SurfaceInductanceCoeffs.Barr
    vertical_field_mutual_inductance = calc_vertical_field_mutual_inductance(0.3, 1.7, coeffs).item()
    np.testing.assert_allclose(vertical_field_mutual_inductance, 1.027670685052496, rtol=5e-2, atol=0)


def test_calc_vertical_magnetic_field():
    coeffs = SurfaceInductanceCoeffs.Barr
    external_inductance = calc_external_inductance(
        0.3 * ureg.dimensionless, 1.7 * ureg.dimensionless, 0.3 * ureg.dimensionless, 1.85 * ureg.m, 0.8 * ureg.dimensionless, coeffs
    )
    vertical_magnetic_field_eq = VertMagneticFieldEq.Barr
    vertical_magnetic_field = calc_vertical_magnetic_field(
        0.3, 1.7, 0.3, 0.8, external_inductance, 1.85 * ureg.m, 8.7e6 * ureg.A, vertical_magnetic_field_eq, coeffs
    ).item()
    np.testing.assert_allclose(vertical_magnetic_field, 1.1286206938400316 * ureg.T, rtol=5e-2, atol=0)


def test_calc_external_inductance():
    coeffs = SurfaceInductanceCoeffs.Barr
    external_inductance = calc_external_inductance(0.3, 1.7, 0.3, 1.85 * ureg.m, 0.8, coeffs).item()
    np.testing.assert_allclose(external_inductance, 2.78305325971087e-06 * ureg.henry, rtol=5e-2, atol=0)


def test_calc_invmu_0_dLedR():
    coeffs = SurfaceInductanceCoeffs.Barr
    external_inductance = calc_external_inductance(0.3, 1.7, 0.3, 1.85 * ureg.m, 0.8, coeffs)
    invmu_0_dLedR = calc_invmu_0_dLedR(0.3, 1.7, 0.3, 0.8, external_inductance, 1.85 * ureg.m, coeffs).item()
    np.testing.assert_allclose(invmu_0_dLedR, 2.1999405545602646, rtol=5e-2, atol=0)


def test_calc_fa():
    coeffs = SurfaceInductanceCoeffs.Barr
    np.testing.assert_allclose(calc_fa(0.3, 0.3, 0.8, coeffs), 1.4335756919903129, rtol=5e-2, atol=0)


def test_calc_fa_Sum_Ne():
    coeffs = SurfaceInductanceCoeffs.Barr
    np.testing.assert_allclose(calc_fa_Sum_Ne(0.3, coeffs), 0.024095227744249464, rtol=5e-2, atol=0)


def test_calc_fa_Sums_Na():
    coeffs = SurfaceInductanceCoeffs.Barr
    np.testing.assert_allclose(calc_fa_Sums_Na.__wrapped__(0.3, coeffs)[0], 1.4293250376924287, rtol=5e-2, atol=0)
    np.testing.assert_allclose(calc_fa_Sums_Na.__wrapped__(0.3, coeffs)[1], 4.559771647300995, rtol=5e-2, atol=0)


def test_calc_fb():
    coeffs = SurfaceInductanceCoeffs.Barr
    np.testing.assert_allclose(calc_fb(0.3, coeffs), 0.08132941228621908, rtol=5e-2, atol=0)


def test_calc_fb_Sum_Nb():
    coeffs = SurfaceInductanceCoeffs.Barr
    np.testing.assert_allclose(calc_fb_Sum_Nb(0.3, coeffs), -0.003446225999999998, rtol=5e-2, atol=0)


def test_calc_fc():
    coeffs = SurfaceInductanceCoeffs.Barr
    np.testing.assert_allclose(calc_fc(0.3, coeffs), 0.970874542, rtol=5e-2, atol=0)


def test_calc_fc_Sum_Nc():
    coeffs = SurfaceInductanceCoeffs.Barr
    np.testing.assert_allclose(calc_fc_Sum_Nc(0.3, coeffs).item(), -0.029125457999999996, rtol=5e-2, atol=0)


def test_calc_fd():
    coeffs = SurfaceInductanceCoeffs.Barr
    np.testing.assert_allclose(calc_fd(0.3, 1.7, coeffs).item(), 0.000826722, rtol=5e-2, atol=0)


def test_calc_fd_Sum_Nd():
    coeffs = SurfaceInductanceCoeffs.Barr
    np.testing.assert_allclose(calc_fd_Sum_Nd(0.3, coeffs).item(), -0.08141999999999998, rtol=5e-2, atol=0)


def test_calc_fg():
    coeffs = SurfaceInductanceCoeffs.Barr
    np.testing.assert_allclose(calc_fg(0.3, 0.3, 0.8, coeffs).item(), -3.2891718339522056, rtol=5e-2, atol=0)


def test_calc_fg_Sum_Ne():
    coeffs = SurfaceInductanceCoeffs.Barr
    np.testing.assert_allclose(calc_fg_Sum_Ce(0.3, coeffs).item(), 0.15044119670967493, rtol=5e-2, atol=0)


def test_calc_fg_Sums_Na():
    coeffs = SurfaceInductanceCoeffs.Barr
    np.testing.assert_allclose(calc_fg_Sums_Na.__wrapped__(0.3, coeffs)[0], 3.4517083961540482, rtol=5e-2, atol=0)
    np.testing.assert_allclose(calc_fg_Sums_Na.__wrapped__(0.3, coeffs)[1], 11.39453620447642, rtol=5e-2, atol=0)


def test_calc_fh():
    coeffs = SurfaceInductanceCoeffs.Barr
    np.testing.assert_allclose(calc_fh(0.3, 1.7, coeffs), -0.77315961077771, rtol=5e-2, atol=0)


def test_calc_fh_Sum_Cb():
    coeffs = SurfaceInductanceCoeffs.Barr
    np.testing.assert_allclose(calc_fh_Sum_Cb(0.3, coeffs), -0.024597188999999988, rtol=5e-2, atol=0)


def test_functionals_against_hirshman():
    epsilon = np.arange(0.1, 1, 0.1)
    kappa = 0  # N/A for Hirshman
    coeffs = SurfaceInductanceCoeffs.Hirshman

    assert np.allclose(
        calc_fa(epsilon, 0, 0, coeffs).values, np.array([2.975, 2.285, 1.848, 1.507, 1.217, 0.957, 0.716, 0.487, 0.272]), rtol=5e-2
    )
    assert np.allclose(
        calc_fb(epsilon, coeffs).values, np.array([0.228, 0.325, 0.403, 0.465, 0.512, 0.542, 0.553, 0.538, 0.508]), rtol=5e-2
    )
    assert np.allclose(
        calc_fc(epsilon, coeffs).values, np.array([1.008, 1.038, 1.093, 1.177, 1.301, 1.486, 1.769, 2.223, 2.864]), rtol=5e-2
    )
    assert np.allclose(
        calc_fd(epsilon, kappa, coeffs).values, np.array([0.022, 0.056, 0.087, 0.113, 0.134, 0.148, 0.155, 0.152, 0.133]), rtol=2.2e-1
    )


def test_inductances_against_hirshman():
    A = np.array([20, 10, 5, 4, 3, 2])
    R0 = A * ureg.m
    coeffs = SurfaceInductanceCoeffs.Hirshman
    kappa = 1.3

    assert all(
        np.isclose(
            calc_vertical_field_mutual_inductance(1 / A, kappa, coeffs).values, np.array([0.99, 0.96, 0.88, 0.83, 0.74, 0.52]), rtol=2e-2
        )
    )
    assert all(
        np.isclose(
            calc_external_inductance(1 / A, kappa, 0, R0, 0, coeffs) / (constants.mu_0 * A),
            np.array([2.93, 2.22, 1.49, 1.26, 0.95, 0.53]),
            rtol=2e-2,
        )
    )


def test_vertical_magnetic_field_against_Eq13():
    """Tests the vertical magnetic field equation derived by Mitarai & Takase (and used by Barr) against a
    typical formulation of the vertical field given in equation 13 in :cite:`mit&taka` at higher
    aspect ratios (where there should be agreement)
    """

    Le_Mitarai = calc_external_inductance(
        1 / np.linspace(1.4, 4), 3, 0.96, 2.8 * np.linspace(1.4, 4) * ureg.m, 0.5, SurfaceInductanceCoeffs.Hirshman
    )
    Bv_Mitarai = calc_vertical_magnetic_field(
        1 / np.linspace(1.4, 4),
        3,
        0.96,
        0.5,
        Le_Mitarai,
        np.linspace(1.4, 4) * 2.8 * ureg.m,
        48e6 * ureg.A,
        VertMagneticFieldEq.Barr,
        SurfaceInductanceCoeffs.Hirshman,
    )

    Bv_Mit13 = calc_vertical_magnetic_field(
        1 / np.linspace(1.4, 4),
        3,
        0.96,
        0.5,
        Le_Mitarai,
        np.linspace(1.4, 4) * 2.8 * ureg.m,
        48e6 * ureg.A,
        VertMagneticFieldEq.Mit_and_Taka_Eq13,
    )
    np.testing.assert_allclose(Bv_Mitarai[17:-1], Bv_Mit13[17:-1], rtol=2e-2)
