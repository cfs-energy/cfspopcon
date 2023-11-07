import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cfspopcon
from scipy import constants  # type: ignore
import pandas as pd

from cfspopcon.unit_handling import ureg, wraps_ufunc
from cfspopcon.unit_handling import Quantity, convert_to_default_units, ureg
from cfspopcon.formulas import calc_plasma_current

# SPARCV1D
RSPARCV1 = Quantity(1.845, ureg.m)
aSPARCV1 = Quantity(0.565, ureg.m)
deltaSPARCV1 = Quantity(0.55, ureg.dimensionless)
kappaSPARCV1 = Quantity(1.7, ureg.dimensionless)
BSPARCV1 = Quantity(12.3, ureg.T)
q95SPARCV1 = Quantity(3.05, ureg.dimensionless)
# tFlatSPARCV1 = 10

ASPARCV1 = Quantity(RSPARCV1 / aSPARCV1, ureg.dimensionless)
epsilonSPARCV1 = Quantity(aSPARCV1 / RSPARCV1, ureg.dimensionless)
# mu0=(4e-7)*np.pi

fSPARCV1 = Quantity((1 + kappaSPARCV1**2 * (1 + 2 * deltaSPARCV1**2 - 1.2 * deltaSPARCV1**3)) / 2.0, ureg.dimensionless)

# IpSPARCV1 = calc_plasma_current(
#     BSPARCV1 ,
#     RSPARCV1 ,
#     epsilonSPARCV1 ,
#     q95SPARCV1 ,
#     fSPARCV1 )

IpSPARCV1 = Quantity(8.7, ureg.MA)

# SPARCV0D
RSPARCV0 = Quantity(1.65, ureg.m)
aSPARCV0 = Quantity(0.56, ureg.m)
deltaSPARCV0 = Quantity(0.32, ureg.dimensionless)
kappaSPARCV0 = Quantity(1.8, ureg.dimensionless)
BSPARCV0 = Quantity(12, ureg.T)
q95SPARCV0 = Quantity(3.02, ureg.dimensionless)
# tFlatSPARCV0 = 10

ASPARCV0 = Quantity(RSPARCV0 / aSPARCV0, ureg.dimensionless)
epsilonSPARCV0 = Quantity(aSPARCV0 / RSPARCV0, ureg.dimensionless)
# mu0=(4e-7)*np.pi

fSPARCV0 = Quantity((1 + kappaSPARCV0**2 * (1 + 2 * deltaSPARCV0**2 - 1.2 * deltaSPARCV0**3)) / 2.0, ureg.dimensionless)

# IpSPARCV0 = calc_plasma_current(
#     BSPARCV0,
#     RSPARCV0,
#     epsilonSPARCV0,
#     q95SPARCV0,
#     fSPARCV0) # gives high value

IpSPARCV0 = Quantity(7.5, ureg.MA)

# JT60-SA
RJT = Quantity(3.06, ureg.m)
aJT = Quantity(1.15, ureg.m)
deltaJT = Quantity(0.45, ureg.dimensionless)
kappaJT = Quantity(1.76, ureg.dimensionless)
BJT = Quantity(2.68, ureg.T)
q95JT = Quantity(3.05, ureg.dimensionless)
# tFlatJT = Quantity(100, ureg.sec)

AJT = Quantity(RJT / aJT, ureg.dimensionless)
epsilonJT = Quantity(aJT / RJT, ureg.dimensionless)
# mu0=(4e-7)*np.pi

fJT = (1 + kappaJT**2 * (1 + 2 * deltaJT**2 - 1.2 * deltaJT**3)) / 2.0

# IpJT = calc_plasma_current(
#     BJT,
#     RJT,
#     epsilonJT,
#     q95JT,
#     fJT)

IpJT = Quantity(5.5, ureg.MA)

# DIII-D
RDIII = Quantity(1.66, ureg.m)
aDIII = Quantity(0.67, ureg.m)
deltaDIII = Quantity(0.45, ureg.dimensionless)
kappaDIII = Quantity(1.76, ureg.dimensionless)
BDIII = Quantity(2.6, ureg.T)
q95DIII = Quantity(3.05, ureg.dimensionless)
# tFlatDIII = Quantity(10, ureg.sec)

ADIII = Quantity(RDIII / aDIII, ureg.dimensionless)
epsilonDIII = Quantity(aDIII / RDIII, ureg.dimensionless)
# mu0=(4e-7)*np.pi

fDIII = Quantity((1 + kappaDIII**2 * (1 + 2 * deltaDIII**2 - 1.2 * deltaDIII**3)) / 2.0, ureg.dimensionless)

# IpDIII = calc_plasma_current(
#     BDIII,
#     RDIII,
#     epsilonDIII,
#     q95DIII,
#     fDIII)

# IpDIII=3.5e6
IpDIII = Quantity(
    1.2, ureg.MA
)  # https://escholarship.org/content/qt78k0v04v/qt78k0v04v_noSplash_c44c701847deffab65024dd9ceff9c59.pdf?t=p15pc5

# ITER
RIT = Quantity(6.2, ureg.m)
aIT = Quantity(2.0, ureg.m)
deltaIT = Quantity(0.33, ureg.dimensionless)
kappaIT = Quantity(1.70, ureg.dimensionless)
BIT = Quantity(5.3, ureg.T)
q95IT = Quantity(3.05, ureg.dimensionless)
# tFlatIT = 500

AIT = Quantity(RIT / aIT, ureg.dimensionless)
epsilonIT = Quantity(aIT / RIT, ureg.dimensionless)
# mu0=(4e-7)*np.pi

fIT = Quantity((1 + kappaIT**2 * (1 + 2 * deltaIT**2 - 1.2 * deltaIT**3)) / 2.0, ureg.dimensionless)

# IpIT = calc_plasma_current(
#     BIT,
#     RIT,
#     epsilonIT,
#     q95IT,
#     fIT)

IpIT = Quantity(15.0, ureg.MA)

# KSTAR
RK = Quantity(1.8, ureg.m)
aK = Quantity(0.5, ureg.m)
deltaK = Quantity(0.8, ureg.dimensionless)
kappaK = Quantity(2.0, ureg.dimensionless)
BK = Quantity(3.5, ureg.T)
q95K = Quantity(3.05, ureg.dimensionless)
# tFlat = 20

AK = Quantity(RK / aK, ureg.dimensionless)
epsilonK = Quantity(aK / RK, ureg.dimensionless)
# mu0=(4e-7)*np.pi

fK = Quantity((1 + kappaK**2 * (1 + 2 * deltaK**2 - 1.2 * deltaK**3)) / 2.0, ureg.dimensionless)

# IpK = calc_plasma_current(
#     BK,
#     RK,
#     epsilonK,
#     q95K,
#     fK)

IpK = Quantity(2.0, ureg.MA)  # https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=585f5eb3f62f3bd76f3d667c1df357562f54c084

# FIRE
RFIRE = Quantity(2.0, ureg.m)
aFIRE = Quantity(0.525, ureg.m)
deltaFIRE = Quantity(0.4, ureg.dimensionless)
kappaFIRE = Quantity(1.8, ureg.dimensionless)
BFIRE = Quantity(10.0, ureg.T)
q95FIRE = Quantity(3.0, ureg.dimensionless)
# tFlat = 20

AFIRE = Quantity(RFIRE / aFIRE, ureg.dimensionless)
epsilonFIRE = Quantity(aFIRE / RFIRE, ureg.dimensionless)
# 3mu0=(4e-7)*np.pi

fFIRE = Quantity((1 + kappaFIRE**2 * (1 + 2 * deltaFIRE**2 - 1.2 * deltaFIRE**3)) / 2.0, ureg.dimensionless)

# IpFIRE = calc_plasma_current(
#     BFIRE,
#     RFIRE,
#     epsilonFIRE,
#     q95FIRE,
#     fFIRE)

# IpFIRE = Quantity(6.44, ureg.MA)
IpFIRE = Quantity(6.5, ureg.MA)  # https://fire.pppl.gov/Snowmass_BP/FIRE.pdf

# ASDEX Upgrade
RASDEX = Quantity(1.65, ureg.m)
aASDEX = Quantity(0.5, ureg.m)
deltaASDEX = Quantity(0.4, ureg.dimensionless)
kappaASDEX = Quantity(1.8, ureg.dimensionless)
BASDEX = Quantity(2.5, ureg.T)
q95ASDEX = Quantity(3.0, ureg.dimensionless)
# tFlat = 8

AASDEX = Quantity(RASDEX / aASDEX, ureg.dimensionless)
epsilonASDEX = Quantity(aASDEX / RASDEX, ureg.dimensionless)

fASDEX = Quantity((1 + kappaASDEX**2 * (1 + 2 * deltaASDEX**2 - 1.2 * deltaASDEX**3)) / 2.0, ureg.dimensionless)

# IpASDEX = calc_plasma_current(
#     BASDEX,
#     RASDEX,
#     epsilonASDEX,
#     q95ASDEX,
#     fASDEX)

# Ip=6.44e6
IpASDEX = Quantity(1.4, ureg.MA)  # https://www.ipp.mpg.de/16208/einfuehrung

# JET
RJET = Quantity(2.96, ureg.m)
aJET = Quantity(1.25, ureg.m)
deltaJET = Quantity(0.45, ureg.dimensionless)
kappaJET = Quantity(1.68, ureg.dimensionless)
BJET = Quantity(3.6, ureg.T)
q95JET = Quantity(3.05, ureg.dimensionless)
# tFlat = 20

AJET = Quantity(RJET / aJET, ureg.dimensionless)
epsilonJET = Quantity(aJET / RJET, ureg.dimensionless)
# mu0=(4e-7)*np.pi

fJET = Quantity((1 + kappaJET**2 * (1 + 2 * deltaJET**2 - 1.2 * deltaJET**3)) / 2.0, ureg.dimensionless)

# IpJET = calc_plasma_current(
#     BJET,
#     RJET,
#     epsilonJET,
#     q95JET,
#     fJET)

# Ip=4.8e6
IpJET = Quantity(5, ureg.MA)  # https://www.ipp.mpg.de/16701/jet

# EAST
REAST = Quantity(1.85, ureg.m)
aEAST = Quantity(0.45, ureg.m)
deltaEAST = Quantity(0.8, ureg.dimensionless)
kappaEAST = Quantity(2.0, ureg.dimensionless)
BEAST = Quantity(3.5, ureg.T)
q95EAST = Quantity(3.05, ureg.dimensionless)
# tFlat = 100

AEAST = Quantity(REAST / aEAST, ureg.dimensionless)
epsilonEAST = Quantity(aEAST / REAST, ureg.dimensionless)

fEAST = Quantity((1 + kappaEAST**2 * (1 + 2 * deltaEAST**2 - 1.2 * deltaEAST**3)) / 2.0, ureg.dimensionless)

# IpEAST = calc_plasma_current(
#     BEAST,
#     REAST,
#     epsilonEAST,
#     q95EAST,
#     fEAST)

IpEAST = Quantity(1.0, ureg.MA)  # https://iopscience.iop.org/article/10.1088/1009-0630/13/1/01

# Alcator C-Mod
RALC = Quantity(0.64, ureg.m)
aALC = Quantity(0.21, ureg.m)
deltaALC = Quantity(0.5, ureg.dimensionless)
kappaALC = Quantity(1.8, ureg.dimensionless)
BALC = Quantity(9.5, ureg.T)
q95ALC = Quantity(3.05, ureg.dimensionless)
# tFlat = 1

AALC = Quantity(RALC / aALC, ureg.dimensionless)
epsilonALC = Quantity(aALC / RALC, ureg.dimensionless)
# mu0=(4e-7)*np.pi

fALC = Quantity((1 + kappaALC**2 * (1 + 2 * deltaALC**2 - 1.2 * deltaALC**3)) / 2.0, ureg.dimensionless)

# IpALC = calc_plasma_current(
#     BALC,
#     RALC,
#     epsilonALC,
#     q95ALC,
#     fALC)

# Ip=3.7e6
IpALC = Quantity(2.02, ureg.MA)  # https://www-internal.psfc.mit.edu/research/alcator/data/fst_cmod.pdf

#### CREATE MACHINE NAMES ###

machines = dict(machine=["CMOD", "AUG", "DIIID", "EAST", "FIRE", "ITER", "JET", "JT60", "KSTAR", "SPARCV1", "SPARCV0"])

machine_R = xr.DataArray(np.stack([RALC, RASDEX, RDIII, REAST, RFIRE, RIT, RJET, RJT, RK, RSPARCV1, RSPARCV0]), coords=machines)

machine_a = xr.DataArray(np.stack([aALC, aASDEX, aDIII, aEAST, aFIRE, aIT, aJET, aJT, aK, aSPARCV1, aSPARCV0]), coords=machines)

machine_delta = xr.DataArray(
    np.stack([deltaALC, deltaASDEX, deltaDIII, deltaEAST, deltaFIRE, deltaIT, deltaJET, deltaJT, deltaK, deltaSPARCV1, deltaSPARCV0]),
    coords=machines,
)

machine_kappa = xr.DataArray(
    np.stack([kappaALC, kappaASDEX, kappaDIII, kappaEAST, kappaFIRE, kappaIT, kappaJET, kappaJT, kappaK, kappaSPARCV1, kappaSPARCV0]),
    coords=machines,
)

machine_B = xr.DataArray(np.stack([BALC, BASDEX, BDIII, BEAST, BFIRE, BIT, BJET, BJT, BK, BSPARCV1, BSPARCV0]), coords=machines)

machine_q95 = xr.DataArray(
    np.stack([q95ALC, q95ASDEX, q95DIII, q95EAST, q95FIRE, q95IT, q95JET, q95JT, q95K, q95SPARCV1, q95SPARCV0]), coords=machines
)

machine_epsilon = xr.DataArray(
    np.stack(
        [
            epsilonALC,
            epsilonASDEX,
            epsilonDIII,
            epsilonEAST,
            epsilonFIRE,
            epsilonIT,
            epsilonJET,
            epsilonJT,
            epsilonK,
            epsilonSPARCV1,
            epsilonSPARCV0,
        ]
    ),
    coords=machines,
)

machine_Ip = xr.DataArray(np.stack([IpALC, IpASDEX, IpDIII, IpEAST, IpFIRE, IpIT, IpJET, IpJT, IpK, IpSPARCV1, IpSPARCV0]), coords=machines)

machine_params = xr.Dataset()
machine_params["R"] = machine_R
machine_params["a"] = machine_a
machine_params["delta"] = machine_delta
machine_params["kappa"] = machine_kappa
machine_params["B"] = machine_B
machine_params["q95"] = machine_q95
machine_params["epsilon"] = machine_epsilon
machine_params["Ip"] = machine_Ip
