import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import simsopt
import simsopt.geo
import simsopt.field
import os

import bdistrib_util
import bdistrib_io
import subprocess
import surfgen


def analyze_coil_complexity_at_distance(
    plasma_surface: simsopt.geo.SurfaceRZFourier, offset: float
):
    msurf = surfgen.surfgen(plasma_surface, offset)
    assert offset * 2 < plasma_surface.major_radius()
    osurf = surfgen.surfgen(plasma_surface, offset * 2)

    bdistrib_io.write_nescin_file("nescin.msurf", msurf)
    bdistrib_io.write_nescin_file("nescin.osurf", osurf)
    subprocess.check_call(
        [
            "../bdistrib/bdistrib",
            bdistrib_io.write_bdistribin(
                bdistrib_io.write_netcdf(
                    "wout_surfaces_python_generated.nc", plasma_surface
                ),
                geometry_option=3,
                geometry_info={
                    "nescin_filename_middle": "nescin.msurf",
                    "nescin_filename_outer": "nescin.osurf",
                },
            ),
        ]
    )
    subprocess.call(
        ["rm", "*.dat"]
    )  # Delete Debug Logs cause I don't know how to disable them
    subprocess.call(["rm", "quasr_coil_check/*.dat"])


for ID in range(50000, 60000):
    simsopt_path = bdistrib_io.get_file_path(ID)
    if os.path.exists(simsopt_path):
        soptobj = simsopt.load(simsopt_path)

        lcfs = soptobj[0][-1]
        # XYZ tensor fourier -> RZ fourier
        rzf = lcfs.to_RZFourier()
        curves = [coil.curve for coil in soptobj[1]]

        bdistrib_out_path = bdistrib_io.get_file_path(ID, "bdistrib")
        subprocess.check_call(["mkdir", "-p", os.path.dirname(bdistrib_out_path)])
        analyze_coil_complexity_at_distance(rzf, rzf.minor_radius() / 2)
        # Move results to correct directory
        subprocess.check_call(
            ["mv", "bdistrib_out.python_generated.nc", bdistrib_out_path, "-u"]
        )
    else:
        print("Skipping", simsopt_path)
