import simsopt
import simsopt.geo
import simsopt.field
import os

import bdistrib_io
import subprocess
import precompute_surfaces


def bdistrib_for_surfaces(
    plasma_surface: simsopt.geo.SurfaceRZFourier,
    msurf: simsopt.geo.SurfaceRZFourier,
    osurf: simsopt.geo.SurfaceRZFourier,
):
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        min_ID = 0
        max_ID = int(sys.argv[1])
    elif len(sys.argv) == 3:
        min_ID = int(sys.argv[1])
        max_ID = int(sys.argv[2])
    else:
        print("plase supply a (min and) max ID until which to process the files.")

    print("Computing surfaces up to ID", max_ID)
    for i in range(min_ID, max_ID):
        if os.path.exists(bdistrib_io.get_file_path(i, "simsopt")):
            surfaces = precompute_surfaces.cached_get_surfaces(i)

            bdistrib_out_path = bdistrib_io.get_file_path(i, "bdistrib")
            subprocess.check_call(["mkdir", "-p", os.path.dirname(bdistrib_out_path)])
            bdistrib_for_surfaces(
                surfaces["lcfs"], surfaces["middle_surf"], surfaces["coil_surf"]
            )
            # Move results to correct directory
            subprocess.check_call(
                ["mv", "bdistrib_out.python_generated.nc", bdistrib_out_path, "-u"]
            )
        else:
            print("Skipping", i)
