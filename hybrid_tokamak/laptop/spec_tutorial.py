from simsopt import mhd
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_serial_solve, least_squares_mpi_solve
from simsopt import geo
from simsopt.util import MpiPartition
import matplotlib.pyplot as plt
import numpy as np

# import logging
# logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)

mpi = MpiPartition(3)

equil = mhd.Spec(
    "hybrid_tokamak/laptop/nfp2_QA_iota0.4.sp",
    # "../../SPEC/InputFiles/Verification/FreeBoundVMEC/Nv=002.L=8.M=12.n.sp",
    mpi,
    verbose=False,
)
# equil.mpol = 7
# equil.ntor = 7
assert not equil.lib.inputlist.lfreebound, "SPEC must be in Fixed boundary mode"

surf = geo.SurfaceRZFourier.from_vmec_input("hybrid_tokamak/input.hybrid_tokamak")
msurf = geo.SurfaceRZFourier.from_vmec_input(
    "hybrid_tokamak/input.hybrid_tokamak", range="field period"
)
msurf.change_resolution(msurf.mpol, 0)
msurf.scale(1.5)
# equil.boundary = surf
surf = equil.boundary
surf.fix_all()
mmax = 1
nmax = 1
surf.fixed_range(0, mmax, -nmax, nmax, False)
surf.fix("rc(0,0)")  # Major radius
print(equil.dof_names)

from fractions import Fraction


# https://stackoverflow.com/questions/38140872/find-the-simplest-rational-number-between-two-given-rational-numbers
def simplest_between(x: Fraction, y: Fraction) -> Fraction:
    """
    Simplest fraction strictly between fractions x and y.
    """
    if x == y:
        raise ValueError("no fractions between x and y")

    # Reduce to case 0 <= x < y
    x, y = min(x, y), max(x, y)
    if y <= 0:
        return -simplest_between(-y, -x)
    elif x < 0:
        return Fraction(0, 1)

    # Find the simplest fraction in (s/t, u/v)
    s, t, u, v = x.numerator, x.denominator, y.numerator, y.denominator
    a, b, c, d = 1, 0, 0, 1
    while True:
        q = s // t
        s, t, u, v = v, u - q * v, t, s - q * t
        a, b, c, d = b + q * a, a, d + q * c, c
        if t > s:
            return Fraction(a + b, c + d)


def find_simples_fractions(x: Fraction, y: Fraction, levels: int = 0) -> list[Fraction]:
    x, y = min(x, y), max(x, y)
    mid = simplest_between(x, y)
    if levels <= 0:
        return [simplest_between(x, y)]

    return [
        *find_simples_fractions(x, mid, levels - 1),
        mid,
        *find_simples_fractions(mid, y, levels - 1),
    ]


fracs = sorted(
    find_simples_fractions(Fraction(-0.38), Fraction(-0.5), 1),
    key=lambda x: x.denominator,
)
print(fracs)
# Construct the greens residues for the 4 most likely islands. Residue uses Fortran indexing for vol
island_residues = [
    (mhd.Residue(equil, frac.numerator, frac.denominator, vol=1, rtol=1e-6).J, 0, 1)
    for frac in fracs[:1]
]

# desired_volume = 3.33
# term2 = (equil.volume, desired_volume, 1.0)

qs = mhd.QuasisymmetryRatioResidualSpec(
    equil, surfaces=np.linspace(0.1, 1, 6), helicity_m=1, helicity_n=0
)

# vac_well = mhd.VacuumWell(equil, surfaces=np.linspace(0, 1, 11))
prob = LeastSquaresProblem.from_tuples(
    [
        (qs.residuals, 0, 5),
        (surf.aspect_ratio, 7, 0.1),
        # (vac_well, 0, 1),
        # *island_residues
    ]
)



# These don't have __self__ and throw an error
def iota_average():
    return np.mean(equil.results.transform.fiota[1, :])
def iota_edge():
    return equil.results.transform.fiota[1, -1]


# iota = p / q
p = -2
q = 5
residue1 = mhd.Residue(equil, p, q)
residue2 = mhd.Residue(equil, p, q, theta=np.pi)
prob = LeastSquaresProblem.from_tuples(
    [
        (surf.aspect_ratio, 6, 1),
        (equil.iota, 0.39, 1),
        # (iota_edge, 0.42, 1),
        (qs.residuals, 0, 2),
        (residue1.J, 0, 2),
        (residue2.J, 0, 2),
    ]
)


# print(" Initial objective function = ", prob.objective())
if mpi is None:
    least_squares_serial_solve(prob, max_nfev=100)
else:
    least_squares_mpi_solve(prob, mpi, grad=True, max_nfev=100)

# Run the final iteration with a higher poincare resolution


if mpi.proc0_world:
    equil.inputlist.nptrj = 32
    equil.inputlist.nppts = 256
    equil.recompute_bell()
    equil.run()

    print("At the optimum,")
    print(" rc(m=1,n=1) = ", surf.get_rc(1, 1))
    print(" zs(m=1,n=1) = ", surf.get_zs(1, 1))
    print(" volume, according to SPEC    = ", equil.volume())
    print(" volume, according to Surface = ", surf.volume())
    print(
        " volume_current_profile, according to SPEC  = ", equil.volume_current_profile
    )
    print(
        " interface_current_profile, according to SPEC = ",
        equil.interface_current_profile,
    )
    print(" iota on axis = ", equil.iota())
    print(" objective function = ", prob.objective())
    print(" Quasisymmetry Ratio = ", qs[0]())

    equil.results.plot_poincare()
    equil.results.plot_iota(xaxis="s")
    equil.results.plot_kam_surface()
    equil.results.plot_modB(
        np.linspace(-0.999, 1, 32), np.linspace(0, 2 * np.pi, 128), np.linspace(0, 0, 2)
    )
    # equil.results.plot_pressure()

    geo.plot([surf], engine="plotly", close=True)
    plt.show()
