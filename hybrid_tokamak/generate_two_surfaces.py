import simsopt
import simsopt.geo
import numpy as np
from quasr_coil_check import bdistrib_io
from quasr_coil_check import precompute_bdistrib
import subprocess

hybrid_surface = simsopt.geo.SurfaceRZFourier.from_vmec_input(
    "hybrid_tokamak/input.NAS.nv.n4.SS.iota43.Fake-ITER.01"
)
middle_surface = simsopt.geo.SurfaceRZFourier.from_vmec_input(
    "hybrid_tokamak/input.NAS.nv.n4.SS.iota43.Fake-ITER.01"
)
# Called outers instead of outer to align with middle & hybrid <3
outers_surface = simsopt.geo.SurfaceRZFourier.from_vmec_input(
    "hybrid_tokamak/input.NAS.nv.n4.SS.iota43.Fake-ITER.01"
)
middle_surface.change_resolution(middle_surface.mpol, 0)
middle_surface.scale(1.6)
middle_surface.set_rc(0, 0, hybrid_surface.get_rc(0, 0) * 0.9)

outers_surface.change_resolution(outers_surface.mpol, 0)
outers_surface.scale(1.8)
outers_surface.set_rc(0, 0, hybrid_surface.get_rc(0, 0) * 0.9)


precompute_bdistrib.bdistrib_for_surfaces(
    hybrid_surface,
    middle_surface,
    outers_surface,
    cwd="hybrid_tokamak",
)

subprocess.check_call(
    [
        "../bdistrib/bdistrib",
        bdistrib_io.write_bdistribin(
            "wout_NAS.nv.n4.SS.iota43.Fake-ITER.01_000_000000.nc",
            geometry_option=1,
            geometry_info={
                "R0": middle_surface.major_radius(),
                "a_middle": middle_surface.minor_radius(),
                "a_outer": outers_surface.minor_radius(),
            },
            dataset_path="hybrid_tokamak/bdistrib_in.hybrid_tokamak",
        ),
    ],
    cwd="hybrid_tokamak",
)


# import matplotlib.pyplot as plt
# import numpy as np

# simsopt.geo.plot(
#     [
#         hybrid_surface,
#         middle_surface,
#         outers_surface,
#     ],
#     engine="plotly",
# )
# hybrid_surface.plot(close=True, show=False)
# middle_surface.plot(close=True, show=False)
# outers_surface.plot(close=True)
