import simsopt
from simsopt import mhd
from simsopt import configs #configs.zoo.get()

from simsopt import objectives
from simsopt.solve import least_squares_serial_solve, least_squares_mpi_solve
from simsopt import geo
from simsopt.util import MpiPartition
import matplotlib.pyplot as plt
import numpy as np

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

mpi = MpiPartition(4)
equil = mhd.Spec(
    "hybrid_tokamak/laptop/nfp2_QA_iota0.4.sp",
    mpi,
    verbose=False,
    keep_all_files=True
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
mmax = 3
nmax = 3
surf.fix_all()
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
qs = mhd.QuasisymmetryRatioResidualSpec(
    equil, surfaces=np.linspace(0.1, 1, 6), helicity_m=1, helicity_n=0
)

# vac_well = mhd.VacuumWell(equil, surfaces=np.linspace(0, 1, 11))
# prob = objectives.LeastSquaresProblem.from_tuples(
#     [
#         (qs.residuals, 0, 5),
#         (surf.aspect_ratio, 7, 0.1),
#         # (vac_well, 0, 1),
#     ]
# ) + objectives.LeastSquaresProblem.from_tuples(
#     # Construct the greens residues for the 1 most likely islands. Residue uses Fortran indexing for vol
#     (mhd.Residue(equil, frac.numerator, frac.denominator, vol=1, rtol=1e-6).J, 0, 1)
#         for frac in fracs[:1]
# )

# These don't have __self__ and throw an error
def iota_average():
    return np.mean(equil.results.transform.fiota[1, :])

def iota_edge(equil):
    equil.run() 
    return equil.results.transform.fiota[1, -1]
iota_edge_target = simsopt.make_optimizable(iota_edge, equil) 


# iota = p / q
p = -2
q = 5
residue1 = mhd.Residue(equil, p, q, s_guess=0.65)
residue2 = mhd.Residue(equil, p, q, s_guess=0.65, theta=np.pi)
prob = objectives.LeastSquaresProblem.from_tuples(
    [
        (qs.residuals, 0, 5),
        (equil.vacuum_well, 1e-2, 1),
        (surf.aspect_ratio, 6, 1),
        (equil.iota, 0.39, 1),
        (iota_edge_target.J, 0.41, 1),
        (residue1.J, 0, 1),
        (residue2.J, 0, 1),
    ]
)


print(prob.dof_names)
print(" Initial objective function = ", prob.objective())
print("   Initial Quasisymmetry Ratio = ", qs.total())
print("   Initial Vacuum Well = ", equil.vacuum_well())
print("   Initial Surface Aspect Ratio = ", surf.aspect_ratio())
print("   Initial Equilibrium Iota = ", equil.iota())
print("   Initial Iota Edge Target = ", iota_edge_target.J())
print("   Initial Residue 1 = ", residue1.J())
print("   Initial Residue 2 = ", residue2.J())

if mpi is None:
    least_squares_serial_solve(prob, max_nfev=100)
else:
    least_squares_mpi_solve(prob, mpi, grad=True, max_nfev=100)

# Run the final iteration with a higher poincare resolution

# equil = mhd.Spec("hybrid_tokamak/laptop/nfp2_QA_iota0.4_000_000082.sp", mpi)

qs = mhd.QuasisymmetryRatioResidualSpec(
    equil, surfaces=np.linspace(0.1, 1, 6), helicity_m=1, helicity_n=0
)
if mpi.proc0_world:
    equil.inputlist.nptrj = 8
    equil.inputlist.nppts = 128
    equil.recompute_bell()
    equil.run()
    
    print(" Final objective function = ", prob.objective())
    print("   Final Quasisymmetry Ratio = ", qs.total())
    print("   Final Vacuum Well = ", equil.vacuum_well())
    print("   Final Surface Aspect Ratio = ", surf.aspect_ratio())
    print("   Final Equilibrium Iota = ", equil.iota())
    print("   Final Iota Edge Target = ", iota_edge_target.J())
    print("   Final Residue 1 = ", residue1.J())
    print("   Final Residue 2 = ", residue2.J())
    
    targets = "_residual_qs_aspect" 
    basename = equil.results.filename.removesuffix(".sp.h5") + targets
    equil.results.plot_poincare()
    plt.savefig( basename+"_poincare.png")
    equil.results.plot_iota(xaxis="s")
    plt.savefig( equil.results.filename+"_iota.png")
    equil.results.plot_kam_surface()
    plt.savefig( equil.results.filename+"_kam.png")
    equil.results.plot_modB(
        np.linspace(-0.999, 1, 32), np.linspace(0, 2 * np.pi, 128), np.linspace(0, 0, 2)
    )
    plt.savefig( equil.results.filename+"_modB.png")
    plt.show()

    import coils_for_QA as coilopt
    coils, J1, J2 =coilopt.optimize(equil.boundary)
    geo.plot([equil.boundary, *coils], engine="plotly", close=True)

