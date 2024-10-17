import simsopt
import simsopt.geo

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
middle_surface.scale(1.5)

outers_surface.change_resolution(outers_surface.mpol, 0)
outers_surface.scale(1.8)


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
