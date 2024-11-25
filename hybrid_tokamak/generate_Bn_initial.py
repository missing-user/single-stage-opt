import simsopt
from simsopt import mhd
import simsopt.geo
import numpy as np
from quasr_coil_check import bdistrib_io
import subprocess

equil = mhd.Spec("hybrid_tokamak/laptop/rotating_ellipse_fb.sp", verbose=True)

hybrid_surface = equil.boundary.copy()
middle_surface = equil.boundary.copy()
# Called outers instead of outer to align with middle & hybrid <3
outers_surface = equil.boundary.copy()

middle_surface.change_resolution(middle_surface.mpol, 0)
middle_surface.scale(1.6)
middle_surface.set_rc(0, 0, hybrid_surface.get_rc(0, 0) )

outers_surface.change_resolution(outers_surface.mpol, 0)
outers_surface.scale(1.8)
outers_surface.set_rc(0, 0, hybrid_surface.get_rc(0, 0) )

simsopt.geo.plot(
    [
        hybrid_surface,
        middle_surface,
        outers_surface,
    ],
    engine="plotly",
)

bdistrib_io.write_netcdf("hybrid_tokamak/wout_nfp2_QA_iota0.4.nc", hybrid_surface)
bdistrib_io.write_nescin_file("hybrid_tokamak/nescin.msurf", middle_surface)
bdistrib_io.write_nescin_file("hybrid_tokamak/nescin.osurf", outers_surface)

subprocess.check_call(
    [
        "../../regcoil/regcoil",
        "regcoil_in.hybrid_tokamak",
    ],
    cwd="hybrid_tokamak",
)


# import matplotlib.pyplot as plt
# import numpy as np

# hybrid_surface.plot(close=True, show=False)
# middle_surface.plot(close=True, show=False)
# outers_surface.plot(close=True)
