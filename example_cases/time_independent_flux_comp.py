import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from cfspopcon.unit_handling import ureg, Quantity, wraps_ufunc, Unitfull, convert_units

# import cfspopcon.formulas.fluxes
from scipy import constants  # type: ignore[import]
import cfspopcon.named_options
from cfspopcon.algorithms.inductances import calc_inductances

from cfspopcon.algorithms import get_algorithm
from cfspopcon.named_options import Algorithms, InternalInductanceGeometry, SurfaceInductanceCoeffs, VertMagneticFieldEq
from cfspopcon.formulas import calc_vertical_magnetic_field, calc_flux_PF, calc_vertical_field_mutual_inductance, calc_invmu_0_dLedR
from MachineParams import machine_params, machines


def calc_flux_rampup_Sugihara(kappa: float, R: float, a: float, Ip: float) -> float:
    A = R.values / a.values
    flux_Ind = constants.mu_0 * A * a.values * (np.log(8 * A) - 1.6) * (Ip.values * 1e6)
    flux_Res = 3.78e-6 * (kappa ** (0.2)) * ((R.values) / ((a.values) ** 0.8)) * ((Ip.values * 1e6) ** 0.8)
    return flux_Res + flux_Ind


def calc_flux_res_rampup_Sugihara(kappa: float, R: float, a: float, Ip: float) -> float:
    flux_Res = 3.78e-6 * (kappa ** (0.2)) * ((R.values) / ((a.values) ** 0.8)) * ((Ip.values * 1e6) ** 0.8)
    # A = R.values / a.values
    # flux_Res = constants.mu_0 * A * a.values * (np.log(8 * A) - 1.6) * (Ip.values * 1e6)
    return flux_Res


def calc_flux_rampup_ROT(R: float, Ip_MA: float) -> float:
    return 2 * R * Ip_MA


dataset = xr.Dataset()
flux_methods = dict(flux_method=["Hirsh&MitTaka13", "Hirsh&Barr", "Hirsh&Jean", "Hirsh&MFEF", "Barr&Barr", "Mit_Mit"])

external_flux_method_dict = {
    "Hirsh&MitTaka13": (SurfaceInductanceCoeffs.Hirshman, VertMagneticFieldEq.Mit_and_Taka_Eq13),
    "Hirsh&Barr": (SurfaceInductanceCoeffs.Hirshman, VertMagneticFieldEq.Barr),
    "Hirsh&Jean": (SurfaceInductanceCoeffs.Hirshman, VertMagneticFieldEq.Jean),
    "Hirsh&MFEF": (SurfaceInductanceCoeffs.Hirshman, VertMagneticFieldEq.MgnticFsionEnrgyFrmlry),
    "Barr&Barr": (SurfaceInductanceCoeffs.Barr, VertMagneticFieldEq.Barr),
    "ROT": (SurfaceInductanceCoeffs.Hirshman, VertMagneticFieldEq.MgnticFsionEnrgyFrmlry),
    "Sugihara": (SurfaceInductanceCoeffs.Barr, VertMagneticFieldEq.Barr),
}

flux_methods_over_machines = xr.DataArray(
    dims=["machine", "flux_method"],
    coords=dict(
        machine=["CMOD", "AUG", "DIIID", "EAST", "FIRE", "ITER", "JET", "JT60", "KSTAR", "SPARCV1", "SPARCV0"],
        flux_method=["Hirsh&MitTaka13", "Hirsh&Barr", "Hirsh&Jean", "Hirsh&MFEF", "Barr&Barr", "ROT", "Sugihara"],
    ),
)
flux_methods_over_machines = flux_methods_over_machines.pint.quantify("weber")

vert_mag_field_over_machines = flux_methods_over_machines.copy(deep=True).pint.dequantify()
vert_mag_field_over_machines = vert_mag_field_over_machines.pint.quantify("T")
PF_flux_over_machines = flux_methods_over_machines.copy(deep=True)
Mv_over_machines = xr.DataArray(
    dims=["machine", "flux_method"],
    coords=dict(
        machine=["CMOD", "AUG", "DIIID", "EAST", "FIRE", "ITER", "JET", "JT60", "KSTAR", "SPARCV1", "SPARCV0"],
        flux_method=["Hirsh&Barr", "Barr&Barr"],
    ),
)

# poloidal_circumference_eqs_over_SPARCV1 = xr.DataArray(dims=["poloidal_circumference_eq"], coords=dict(poloidal_circumference_eq=[
#         "ITER",
#         "LpBasic",
#         "LpNew"
# ]))

machine_name = machines["machine"]
Sugihara_flux_res = xr.DataArray(
    dims=["machine"], coords=dict(machine=["CMOD", "AUG", "DIIID", "EAST", "FIRE", "ITER", "JET", "JT60", "KSTAR", "SPARCV1", "SPARCV0"])
)
Ejima_flux_res = Sugihara_flux_res.copy(deep=True)

###***BETA_POLOIDAL***###
dataset["beta_poloidal"] = 0.1

###***TODO: INTERNAL_INDUCTANCE***###
dataset["q_0"] = 1
dataset["internal_inductance_geometry"] = InternalInductanceGeometry.Cylindrical

###***Flattop LOOP_VOLTAGE***###
dataset["loop_voltage"] = 0.5 * ureg.volts  # N/A for this calc since not considering flattop

###***Not-Applicable Geometry Variables***###
dataset["elongation_ratio_sep_to_areal"] = 1
dataset["triangularity_ratio_sep_to_psi95"] = 1

###***FLUX VARIABLES THAT ARE SET THE SAME FOR EVERY MACHINE***###
dataset["ejima_coefficient"] = 0.45
dataset["total_flux_available_from_CS"] = 35.2 * ureg.weber  # N/A for these calcs

dataset["custom_internal_inductivity"] = False
#dataset["internal_inductivity"] = 0.91

import warnings

warnings.filterwarnings("ignore")  # ignore unit strip warning happening with Sugihara and ROT functions
###***RUN CALCULATIONS***###
for machine_name in machines["machine"]:
    for external_flux_method in external_flux_method_dict:
        if external_flux_method != "ROT" and external_flux_method != "Sugihara":
            dataset["magnetic_field_on_axis"] = machine_params.sel(machine=machine_name).B
            dataset["major_radius"] = machine_params.sel(machine=machine_name).R
            dataset["minor_radius"] = machine_params.sel(machine=machine_name).a
            dataset["triangularity_psi95"] = machine_params.sel(machine=machine_name).delta
            dataset["areal_elongation"] = machine_params.sel(machine=machine_name).kappa
            dataset["inverse_aspect_ratio"] = machine_params.sel(machine=machine_name).epsilon
            dataset["plasma_current"] = machine_params.sel(machine=machine_name).Ip
            dataset["surface_inductance_coefficients"] = external_flux_method_dict[external_flux_method][0]
            dataset["vertical_magnetic_field_eq"] = external_flux_method_dict[external_flux_method][1]
            # if machine_name == "SPARCV1": # Test various poloidal circumference equations for SPARCV1
            #      for eq in PoloidalCircumferenceEq:
            #           dataset["poloidal_circumference_eq"] = eq
            #           get_algorithm(Algorithms.calc_geometry).update_dataset(dataset, in_place=True)
            #           get_algorithm(Algorithms.calc_inductances).update_dataset(dataset, in_place=True)
            #           poloidal_circumference_eqs_over_SPARCV1.loc[dict(poloidal_circumference_eq=eq.name)] = dataset.get("internal_inductance")
            # dataset["poloidal_circumference_eq"] = poloidal_circumference_eq_all_machines
            get_algorithm(Algorithms.calc_geometry).update_dataset(dataset, in_place=True)
            get_algorithm(Algorithms.calc_inductances).update_dataset(dataset, in_place=True)
            # dataset["internal_inductance"] = 1.02468e-06 * ureg.henry
            get_algorithm(Algorithms.calc_fluxes).update_dataset(dataset, in_place=True)
            # STORE VALUES
            if machine_name == "SPARCV1":
                # print("HERE")
                SPARCV1_external_flux = dataset.get("external_flux").item()
                SPARCV1_internal_flux = dataset.get("internal_flux").item()
                SPARCV1_resistive_flux = dataset.get("resistive_flux").item()
                SPARCV1_PF_flux = dataset.get("PF_flux").item()
                print(
                    external_flux_method,
                    ": ",
                    "SPARCV1_resistive_flux ",
                    SPARCV1_resistive_flux,
                    "SPARCV1_external_flux ",
                    SPARCV1_external_flux,
                    "SPARCV1_internal_flux ",
                    SPARCV1_internal_flux,
                    "SPARCV1_PF_flux ",
                    SPARCV1_PF_flux,
                    "flux_needed_from_CS_for_ramp",
                    dataset.get("flux_needed_from_CS_over_rampup").item(),
                )
            PF_flux_over_machines.loc[dict(machine=machine_name, flux_method=external_flux_method)] = dataset["PF_flux"]
            flux_methods_over_machines.loc[dict(machine=machine_name, flux_method=external_flux_method)] = dataset[
                "flux_needed_from_CS_over_rampup"
            ]  # + dataset["PF_flux"] # add in to compare flux consumed
            vert_mag_field_over_machines.loc[dict(machine=machine_name, flux_method=external_flux_method)] = dataset[
                "vertical_magnetic_field"
            ]
            if external_flux_method == "Hirsh&Barr" or external_flux_method == "Barr&Barr":
                Mv_over_machines.loc[dict(machine=machine_name, flux_method=external_flux_method)] = dataset[
                    "vertical_field_mutual_inductance"
                ]
            Ejima_flux_res.loc[dict(machine=machine_name)] = dataset["resistive_flux"]
            # Ejima_flux_res.loc[dict(machine=machine_name)] = dataset["internal_flux"] + dataset["external_flux"]
        elif external_flux_method == "ROT":
            ROT_flux = calc_flux_rampup_ROT(dataset.get("major_radius"), dataset.get("plasma_current"))
            flux_methods_over_machines.loc[dict(machine=machine_name, flux_method=external_flux_method)] = (
                ROT_flux.pint.dequantify() * ureg.weber
            )
        elif external_flux_method == "Sugihara":
            Sugihara_flux = calc_flux_rampup_Sugihara(
                dataset.get("areal_elongation"), dataset.get("major_radius"), dataset.get("minor_radius"), dataset.get("plasma_current")
            )
            Sugihara_flux_res.loc[dict(machine=machine_name)] = calc_flux_res_rampup_Sugihara(
                dataset.get("areal_elongation"), dataset.get("major_radius"), dataset.get("minor_radius"), dataset.get("plasma_current")
            )
            flux_methods_over_machines.loc[dict(machine=machine_name, flux_method=external_flux_method)] = Sugihara_flux * ureg.weber

### Relative tolerance/difference function ###
def calc_reltol(other, poynt):
    reltol = np.zeros(len(other))
    i = 0

    while i < len(other):
        reltol[i] = "{:g}".format(float("{:.{p}g}".format(100 * np.abs(other[i] - poynt[i]) / other[i], p=3)))
        i = i + 1

    return reltol


###***Percent Differnce Between Ejima and Sugihara Resistive***###
# breakpoint()
machine_name = machines["machine"]
res_data = pd.DataFrame(
    {
        "Sugihara": Sugihara_flux_res,
        "Ejima": Ejima_flux_res,
    },
    index=machine_name,
)

ax = res_data.plot.bar(
    figsize=(20, 5),
    rot=0,
    width=0.95,
    color={
        "Sugihara": "xkcd:ocean blue",
        "Ejima": "xkcd:pumpkin",
    },
)
for p in ax.patches:
    label = "{:.2g}".format(p.get_height())  # Round to two decimal places
    ax.annotate(
        str(label),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        fontsize=6,
        xytext=(0, 5),
        textcoords="offset points",
        color="black",
        weight="bold",
    )
ax.set_ylabel("Inductively Consumed Flux [Webers]")
ax.set_title("Difference Between Sugihara and Poynting Inductive Flux Calc Methods")
plt.savefig("/Users/isaacsavona/Documents/CFS/FluxConsump/NewPlots/SugRes_v_PoyntRes@RampUp.svg", format="svg")

###***Percent Differnce Between ROT.vs.(Barr&Barr and Hirsh&Barr) and Sugihara.vs.(Barr&Barr and Hirsh&Barr)***###
machine_name = machines["machine"]

reltol_ROTvsHirsh_Barr = calc_reltol(
    flux_methods_over_machines.sel(flux_method="ROT").values, flux_methods_over_machines.sel(flux_method="Hirsh&Barr").values
)
reltol_ROTvsBarr_Barr = calc_reltol(
    flux_methods_over_machines.sel(flux_method="ROT").values, flux_methods_over_machines.sel(flux_method="Barr&Barr").values
)
reltol_SugiharavsHirsh_Barr = calc_reltol(
    flux_methods_over_machines.sel(flux_method="Sugihara").values,
    flux_methods_over_machines.sel(flux_method="Hirsh&Barr").values + PF_flux_over_machines.sel(flux_method="Hirsh&Barr").values,
)
reltol_SugiharavsBarr_Barr = calc_reltol(
    flux_methods_over_machines.sel(flux_method="Sugihara").values,
    flux_methods_over_machines.sel(flux_method="Barr&Barr").values + PF_flux_over_machines.sel(flux_method="Barr&Barr").values,
)

df = pd.DataFrame(
    {
        "ROTvsHirsh&Barr": reltol_ROTvsHirsh_Barr,
        "ROTvsBarr&Barr": reltol_ROTvsBarr_Barr,
        "SugiharavsHirsh&Barr": reltol_SugiharavsHirsh_Barr,
        "SugiharavsBarr&Barr": reltol_SugiharavsBarr_Barr,
    },
    index=machine_name,
)

ax = df.plot.bar(
    figsize=(16, 5),
    rot=0,
    width=1,
    color={
        "ROTvsHirsh&Barr": "xkcd:grass green",
        "ROTvsBarr&Barr": "xkcd:scarlet",
        "SugiharavsHirsh&Barr": "xkcd:ocean blue",
        "SugiharavsBarr&Barr": "xkcd:pumpkin",
    },
)
for p in ax.patches:
    label = "{:.2g}".format(p.get_height())  # Round to two decimal places
    ax.annotate(
        str(label),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        fontsize=7,
        xytext=(0, 5),
        textcoords="offset points",
        color="black",
        weight="bold",
    )
ax.set_ylabel("Relative Difference [%]")
ax.set_title("Difference Between ROT or Sugihara vs Poynting Flux Calc Methods")
ax.legend(loc="lower left")
plt.savefig("/Users/isaacsavona/Documents/CFS/FluxConsump/NewPlots/ROTSug_v_Poynt@RampUp.svg", format="svg")

### SCALE FLUXES FOR CLARITY ###
machine_name = machines["machine"]
scaled_machine_name = machine_name.copy()

scaling_factor_ITER = flux_methods_over_machines.loc[dict(machine="SPARCV1")] / flux_methods_over_machines.loc[dict(machine="ITER")]
scaling_factor_CMOD = flux_methods_over_machines.loc[dict(machine="SPARCV1")] / flux_methods_over_machines.loc[dict(machine="CMOD")]
scaling_factor_AUG = flux_methods_over_machines.loc[dict(machine="SPARCV1")] / flux_methods_over_machines.loc[dict(machine="AUG")]
scaling_factor_DIIID = flux_methods_over_machines.loc[dict(machine="SPARCV1")] / flux_methods_over_machines.loc[dict(machine="DIIID")]
scaling_factor_EAST = flux_methods_over_machines.loc[dict(machine="SPARCV1")] / flux_methods_over_machines.loc[dict(machine="EAST")]
scaling_factor_FIRE = flux_methods_over_machines.loc[dict(machine="SPARCV1")] / flux_methods_over_machines.loc[dict(machine="FIRE")]
scaling_factor_KSTAR = flux_methods_over_machines.loc[dict(machine="SPARCV1")] / flux_methods_over_machines.loc[dict(machine="KSTAR")]

scaled_machine_name[5] = str("{:.2g}".format(scaling_factor_ITER.mean().values)) + "*" + scaled_machine_name[5]
scaled_machine_name[0] = str("{:.2g}".format(scaling_factor_CMOD.mean().values)) + "*" + scaled_machine_name[0]
scaled_machine_name[1] = str("{:.2g}".format(scaling_factor_AUG.mean().values)) + "*" + scaled_machine_name[1]
scaled_machine_name[2] = str("{:.2g}".format(scaling_factor_DIIID.mean().values)) + "*" + scaled_machine_name[2]
scaled_machine_name[3] = str("{:.2g}".format(scaling_factor_EAST.mean().values)) + "*" + scaled_machine_name[3]
scaled_machine_name[4] = str("{:.2g}".format(scaling_factor_FIRE.mean().values)) + "*" + scaled_machine_name[4]
scaled_machine_name[8] = str("{:.2g}".format(scaling_factor_KSTAR.mean().values)) + "*" + scaled_machine_name[8]

flux_methods_over_machines.loc[dict(machine="ITER")] = scaling_factor_ITER.mean() * flux_methods_over_machines.loc[dict(machine="ITER")]
flux_methods_over_machines.loc[dict(machine="CMOD")] = scaling_factor_CMOD.mean() * flux_methods_over_machines.loc[dict(machine="CMOD")]
flux_methods_over_machines.loc[dict(machine="AUG")] = scaling_factor_AUG.mean() * flux_methods_over_machines.loc[dict(machine="AUG")]
flux_methods_over_machines.loc[dict(machine="DIIID")] = scaling_factor_DIIID.mean() * flux_methods_over_machines.loc[dict(machine="DIIID")]
flux_methods_over_machines.loc[dict(machine="EAST")] = scaling_factor_EAST.mean() * flux_methods_over_machines.loc[dict(machine="EAST")]
flux_methods_over_machines.loc[dict(machine="FIRE")] = scaling_factor_FIRE.mean() * flux_methods_over_machines.loc[dict(machine="FIRE")]
flux_methods_over_machines.loc[dict(machine="KSTAR")] = scaling_factor_KSTAR.mean() * flux_methods_over_machines.loc[dict(machine="KSTAR")]
flux_methods_over_machines = flux_methods_over_machines.assign_coords(machine=scaled_machine_name)

###***PLOTTING_needed_CS_flux_over_rampup***###
df = pd.DataFrame(
    {
        "Hirsh&MitTaka13": flux_methods_over_machines.sel(machine=scaled_machine_name, flux_method="Hirsh&MitTaka13"),
        "Hirsh&Barr": flux_methods_over_machines.sel(machine=scaled_machine_name, flux_method="Hirsh&Barr"),
        "Hirsh&Jean": flux_methods_over_machines.sel(machine=scaled_machine_name, flux_method="Hirsh&Jean"),
        "Hirsh&MFEF": flux_methods_over_machines.sel(machine=scaled_machine_name, flux_method="Hirsh&MFEF"),
        "Barr&Barr": flux_methods_over_machines.sel(machine=scaled_machine_name, flux_method="Barr&Barr"),
        "ROT": flux_methods_over_machines.sel(machine=scaled_machine_name, flux_method="ROT"),
        "Sugihara": flux_methods_over_machines.sel(machine=scaled_machine_name, flux_method="Sugihara"),
    },
    index=scaled_machine_name,
)

ax = df.plot.bar(
    figsize=(15, 5),
    rot=0,
    width=0.95,
    color={
        "Hirsh&MitTaka13": "xkcd:grass green",
        "Hirsh&Barr": "xkcd:scarlet",
        "Hirsh&Jean": "xkcd:ocean blue",
        "Hirsh&MFEF": "xkcd:pumpkin",
        "Barr&Barr": "xkcd:purple",
        "ROT": "xkcd:dark yellow",
        "Sugihara": "xkcd:brown",
    },
)
for p in ax.patches:
    label = "{:.2g}".format(p.get_height())  # Round to two decimal places
    ax.annotate(
        str(label),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        fontsize=7,
        xytext=(0, 5),
        textcoords="offset points",
        color="black",
        weight="bold",
    )
ax.set_ylabel("Flux [Weber]")
ax.set_title("Necessary Flux from the CS Over Ramp-Up")
ax.legend(loc="lower left")
plt.savefig("/Users/isaacsavona/Documents/CFS/FluxConsump/NewPlots/ROTSugPoynt@RampUp.svg", format="svg")

##############################################################

### SCALE VERICAL FIELDS FOR CLARITY ###
machine_name = machines["machine"]
scaled_machine_name = machine_name.copy()

scaling_factor_ITER = vert_mag_field_over_machines.loc[dict(machine="SPARCV1")] / vert_mag_field_over_machines.loc[dict(machine="ITER")]
scaling_factor_CMOD = vert_mag_field_over_machines.loc[dict(machine="SPARCV1")] / vert_mag_field_over_machines.loc[dict(machine="CMOD")]
scaling_factor_AUG = vert_mag_field_over_machines.loc[dict(machine="SPARCV1")] / vert_mag_field_over_machines.loc[dict(machine="AUG")]
scaling_factor_DIIID = vert_mag_field_over_machines.loc[dict(machine="SPARCV1")] / vert_mag_field_over_machines.loc[dict(machine="DIIID")]
scaling_factor_EAST = vert_mag_field_over_machines.loc[dict(machine="SPARCV1")] / vert_mag_field_over_machines.loc[dict(machine="EAST")]
scaling_factor_FIRE = vert_mag_field_over_machines.loc[dict(machine="SPARCV1")] / vert_mag_field_over_machines.loc[dict(machine="FIRE")]
scaling_factor_KSTAR = vert_mag_field_over_machines.loc[dict(machine="SPARCV1")] / vert_mag_field_over_machines.loc[dict(machine="KSTAR")]
scaling_factor_JT60 = vert_mag_field_over_machines.loc[dict(machine="SPARCV1")] / vert_mag_field_over_machines.loc[dict(machine="JT60")]
scaling_factor_JET = vert_mag_field_over_machines.loc[dict(machine="SPARCV1")] / vert_mag_field_over_machines.loc[dict(machine="JET")]

scaled_machine_name[5] = str("{:.2g}".format(scaling_factor_ITER.mean().values)) + "*" + scaled_machine_name[5]
scaled_machine_name[0] = str("{:.2g}".format(scaling_factor_CMOD.mean().values)) + "*" + scaled_machine_name[0]
scaled_machine_name[1] = str("{:.2g}".format(scaling_factor_AUG.mean().values)) + "*" + scaled_machine_name[1]
scaled_machine_name[2] = str("{:.2g}".format(scaling_factor_DIIID.mean().values)) + "*" + scaled_machine_name[2]
scaled_machine_name[3] = str("{:.2g}".format(scaling_factor_EAST.mean().values)) + "*" + scaled_machine_name[3]
scaled_machine_name[4] = str("{:.2g}".format(scaling_factor_FIRE.mean().values)) + "*" + scaled_machine_name[4]
scaled_machine_name[8] = str("{:.2g}".format(scaling_factor_KSTAR.mean().values)) + "*" + scaled_machine_name[8]
scaled_machine_name[7] = str("{:.2g}".format(scaling_factor_JT60.mean().values)) + "*" + scaled_machine_name[7]
scaled_machine_name[6] = str("{:.2g}".format(scaling_factor_JET.mean().values)) + "*" + scaled_machine_name[6]

vert_mag_field_over_machines.loc[dict(machine="ITER")] = scaling_factor_ITER.mean() * vert_mag_field_over_machines.loc[dict(machine="ITER")]
vert_mag_field_over_machines.loc[dict(machine="CMOD")] = scaling_factor_CMOD.mean() * vert_mag_field_over_machines.loc[dict(machine="CMOD")]
vert_mag_field_over_machines.loc[dict(machine="AUG")] = scaling_factor_AUG.mean() * vert_mag_field_over_machines.loc[dict(machine="AUG")]
vert_mag_field_over_machines.loc[dict(machine="DIIID")] = (
    scaling_factor_DIIID.mean() * vert_mag_field_over_machines.loc[dict(machine="DIIID")]
)
vert_mag_field_over_machines.loc[dict(machine="EAST")] = scaling_factor_EAST.mean() * vert_mag_field_over_machines.loc[dict(machine="EAST")]
vert_mag_field_over_machines.loc[dict(machine="FIRE")] = scaling_factor_FIRE.mean() * vert_mag_field_over_machines.loc[dict(machine="FIRE")]
vert_mag_field_over_machines.loc[dict(machine="KSTAR")] = (
    scaling_factor_KSTAR.mean() * vert_mag_field_over_machines.loc[dict(machine="KSTAR")]
)
vert_mag_field_over_machines.loc[dict(machine="JT60")] = scaling_factor_JT60.mean() * vert_mag_field_over_machines.loc[dict(machine="JT60")]
vert_mag_field_over_machines.loc[dict(machine="JET")] = scaling_factor_JET.mean() * vert_mag_field_over_machines.loc[dict(machine="JET")]

vert_mag_field_over_machines = vert_mag_field_over_machines.assign_coords(machine=scaled_machine_name)

###***PLOTTING_Vertical_Magnetic_Field***###

# machine_name = machines["machine"]

df = pd.DataFrame(
    {
        "Hirsh&MitTaka13": vert_mag_field_over_machines.sel(machine=scaled_machine_name, flux_method="Hirsh&MitTaka13"),
        "Hirsh&Barr": vert_mag_field_over_machines.sel(machine=scaled_machine_name, flux_method="Hirsh&Barr"),
        "Hirsh&Jean": vert_mag_field_over_machines.sel(machine=scaled_machine_name, flux_method="Hirsh&Jean"),
        "Hirsh&MFEF": vert_mag_field_over_machines.sel(machine=scaled_machine_name, flux_method="Hirsh&MFEF"),
        "Barr&Barr": vert_mag_field_over_machines.sel(machine=scaled_machine_name, flux_method="Barr&Barr"),
    },
    index=scaled_machine_name,
)

ax = df.plot.bar(
    figsize=(20, 5),
    rot=0,
    width=0.95,
    color={
        "Hirsh&MitTaka13": "xkcd:grass green",
        "Hirsh&Barr": "xkcd:scarlet",
        "Hirsh&Jean": "xkcd:ocean blue",
        "Hirsh&MFEF": "xkcd:pumpkin",
        "Barr&Barr": "xkcd:purple",
    },
)
for p in ax.patches:
    label = "{:.2g}".format(p.get_height())  # Round to two decimal places
    ax.annotate(
        str(label),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        fontsize=6,
        xytext=(0, 5),
        textcoords="offset points",
        color="black",
        weight="bold",
    )
ax.set_ylabel("Vertical Magnetic Field [T]")
ax.set_title("Vertical Magnetic Field at the End of Rampup")
ax.legend(loc="lower left")
plt.savefig("/Users/isaacsavona/Documents/CFS/FluxConsump/NewPlots/VertMagField@RampUp.svg", format="svg")
# plt.show()

#########################################################################

### SCALE PF FLUXES FOR CLARITY ###
machine_name = machines["machine"]
scaled_machine_name = machine_name.copy()

scaling_factor_ITER = PF_flux_over_machines.loc[dict(machine="SPARCV1")] / PF_flux_over_machines.loc[dict(machine="ITER")]
scaling_factor_CMOD = PF_flux_over_machines.loc[dict(machine="SPARCV1")] / PF_flux_over_machines.loc[dict(machine="CMOD")]
scaling_factor_AUG = PF_flux_over_machines.loc[dict(machine="SPARCV1")] / PF_flux_over_machines.loc[dict(machine="AUG")]
scaling_factor_DIIID = PF_flux_over_machines.loc[dict(machine="SPARCV1")] / PF_flux_over_machines.loc[dict(machine="DIIID")]
scaling_factor_EAST = PF_flux_over_machines.loc[dict(machine="SPARCV1")] / PF_flux_over_machines.loc[dict(machine="EAST")]
scaling_factor_FIRE = PF_flux_over_machines.loc[dict(machine="SPARCV1")] / PF_flux_over_machines.loc[dict(machine="FIRE")]
scaling_factor_KSTAR = PF_flux_over_machines.loc[dict(machine="SPARCV1")] / PF_flux_over_machines.loc[dict(machine="KSTAR")]
scaling_factor_JT60 = PF_flux_over_machines.loc[dict(machine="SPARCV1")] / PF_flux_over_machines.loc[dict(machine="JT60")]
scaling_factor_JET = PF_flux_over_machines.loc[dict(machine="SPARCV1")] / PF_flux_over_machines.loc[dict(machine="JET")]

scaled_machine_name[5] = str("{:.2g}".format(scaling_factor_ITER.mean().values)) + "*" + scaled_machine_name[5]
scaled_machine_name[0] = str("{:.2g}".format(scaling_factor_CMOD.mean().values)) + "*" + scaled_machine_name[0]
scaled_machine_name[1] = str("{:.2g}".format(scaling_factor_AUG.mean().values)) + "*" + scaled_machine_name[1]
scaled_machine_name[2] = str("{:.2g}".format(scaling_factor_DIIID.mean().values)) + "*" + scaled_machine_name[2]
scaled_machine_name[3] = str("{:.2g}".format(scaling_factor_EAST.mean().values)) + "*" + scaled_machine_name[3]
scaled_machine_name[4] = str("{:.2g}".format(scaling_factor_FIRE.mean().values)) + "*" + scaled_machine_name[4]
scaled_machine_name[8] = str("{:.2g}".format(scaling_factor_KSTAR.mean().values)) + "*" + scaled_machine_name[8]
scaled_machine_name[7] = str("{:.2g}".format(scaling_factor_JT60.mean().values)) + "*" + scaled_machine_name[7]
scaled_machine_name[6] = str("{:.2g}".format(scaling_factor_JET.mean().values)) + "*" + scaled_machine_name[6]

PF_flux_over_machines.loc[dict(machine="ITER")] = scaling_factor_ITER.mean() * PF_flux_over_machines.loc[dict(machine="ITER")]
PF_flux_over_machines.loc[dict(machine="CMOD")] = scaling_factor_CMOD.mean() * PF_flux_over_machines.loc[dict(machine="CMOD")]
PF_flux_over_machines.loc[dict(machine="AUG")] = scaling_factor_AUG.mean() * PF_flux_over_machines.loc[dict(machine="AUG")]
PF_flux_over_machines.loc[dict(machine="DIIID")] = scaling_factor_DIIID.mean() * PF_flux_over_machines.loc[dict(machine="DIIID")]
PF_flux_over_machines.loc[dict(machine="EAST")] = scaling_factor_EAST.mean() * PF_flux_over_machines.loc[dict(machine="EAST")]
PF_flux_over_machines.loc[dict(machine="FIRE")] = scaling_factor_FIRE.mean() * PF_flux_over_machines.loc[dict(machine="FIRE")]
PF_flux_over_machines.loc[dict(machine="KSTAR")] = scaling_factor_KSTAR.mean() * PF_flux_over_machines.loc[dict(machine="KSTAR")]
PF_flux_over_machines.loc[dict(machine="JT60")] = scaling_factor_JT60.mean() * PF_flux_over_machines.loc[dict(machine="JT60")]
PF_flux_over_machines.loc[dict(machine="JET")] = scaling_factor_JET.mean() * PF_flux_over_machines.loc[dict(machine="JET")]

PF_flux_over_machines = PF_flux_over_machines.assign_coords(machine=scaled_machine_name)

###***PLOTTING_PF_Flux***###
df = pd.DataFrame(
    {
        "Hirsh&MitTaka13": PF_flux_over_machines.sel(machine=scaled_machine_name, flux_method="Hirsh&MitTaka13"),
        "Hirsh&Barr": PF_flux_over_machines.sel(machine=scaled_machine_name, flux_method="Hirsh&Barr"),
        "Hirsh&Jean": PF_flux_over_machines.sel(machine=scaled_machine_name, flux_method="Hirsh&Jean"),
        "Hirsh&MFEF": PF_flux_over_machines.sel(machine=scaled_machine_name, flux_method="Hirsh&MFEF"),
        "Barr&Barr": PF_flux_over_machines.sel(machine=scaled_machine_name, flux_method="Barr&Barr"),
    },
    index=scaled_machine_name,
)

ax = df.plot.bar(
    figsize=(15, 5),
    rot=0,
    width=0.95,
    color={
        "Hirsh&MitTaka13": "xkcd:grass green",
        "Hirsh&Barr": "xkcd:scarlet",
        "Hirsh&Jean": "xkcd:ocean blue",
        "Hirsh&MFEF": "xkcd:pumpkin",
        "Barr&Barr": "xkcd:purple",
    },
)
for p in ax.patches:
    label = "{:.2g}".format(p.get_height())  # Round to two decimal places
    ax.annotate(
        str(label),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        fontsize=6,
        xytext=(0, 5),
        textcoords="offset points",
        color="black",
        weight="bold",
    )
ax.set_ylabel("Vertical Field Surface Flux Contribution [Weber]")
ax.set_title("Surface Flux Constribution from the Vertical Field Over Rampup")
ax.legend(loc="lower left")
plt.savefig("/Users/isaacsavona/Documents/CFS/FluxConsump/NewPlots/SurfFluxContribution@RampUp.svg", format="svg")

###################################################################################

###***Show Difference in Mv between Hirshman and Barr***###

df = pd.DataFrame(
    {
        # "Hirsh&MitTaka13": vert_mag_field_over_machines.sel(machine=scaled_machine_name, flux_method="Hirsh&MitTaka13"),
        "Hirsh": Mv_over_machines.sel(machine=machine_name, flux_method="Hirsh&Barr"),
        # "Hirsh&Jean": vert_mag_field_over_machines.sel(machine=scaled_machine_name, flux_method="Hirsh&Jean"),
        # "Hirsh&MFEF": vert_mag_field_over_machines.sel(machine=scaled_machine_name, flux_method="Hirsh&MFEF"),
        "Barr": Mv_over_machines.sel(machine=machine_name, flux_method="Barr&Barr"),
    },
    index=machine_name,
)

ax = df.plot.bar(
    figsize=(20, 5),
    rot=0,
    width=0.95,
    color={
        # "Hirsh&MitTaka13": "xkcd:grass green",
        "Hirsh": "xkcd:scarlet",
        # "Hirsh&Jean": "xkcd:ocean blue",
        # "Hirsh&MFEF": "xkcd:pumpkin",
        "Barr": "xkcd:purple",
    },
)
for p in ax.patches:
    label = "{:.2g}".format(p.get_height())  # Round to two decimal places
    ax.annotate(
        str(label),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        fontsize=6,
        xytext=(0, 5),
        textcoords="offset points",
        color="black",
        weight="bold",
    )
ax.set_ylabel("Vertical Magnetic Field Mutual Inductance [~]")
ax.set_title("Vertical Magnetic Field Mutual Inductance at the End of Ramp Up")
ax.legend(loc="lower left")
plt.savefig("/Users/isaacsavona/Documents/CFS/FluxConsump/NewPlots/VertMagFieldMutInduct@RampUp.svg", format="svg")

###################################################################################

###***Show Differences in Cp for SPARCV1 on Internal Inductance***###

# fig, ax = plt.subplots()
# data = {
# r"$ITER$" : poloidal_circumference_eqs_over_SPARCV1.sel(poloidal_circumference_eq="ITER"),
# r"$L^{Basic}_P$" : poloidal_circumference_eqs_over_SPARCV1.sel(poloidal_circumference_eq="LpBasic"),
# r"$L^{New}_P$"  : poloidal_circumference_eqs_over_SPARCV1.sel(poloidal_circumference_eq="LpNew"),
# }

# fig.set_size_inches(12, 5)
# ax.bar(data.keys(), data.values(), color = ["xkcd:grass green",
# "xkcd:scarlet",
# "xkcd:ocean blue"], label = [r"$ITER$", r'$L^{Basic}_P$', r'$L^{New}_P$'])
# for p in ax.patches:
#     label = '{:.2g}'.format(p.get_height())  # Round to two decimal places
#     ax.annotate(str(label), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=8, xytext=(0, 5), textcoords='offset points', color='black', weight='bold')
# ax.set_ylabel("Internal Inductance [Henry]")
# ax.set_title("Internal Inductance Comparison for Different Poloidal Circumference Formulation")
# ax.legend([r"$ITER$", r'$L^{Basic}_P$', r'$L^{New}_P$'])
# plt.savefig('/Users/isaacsavona/Documents/CFS/FluxConsump/NewPlots/CpEq_v_Inducts@RampUp.svg', format='svg')

plt.show()  # show all plots
