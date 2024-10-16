import subprocess
import os
import simsopt
import simsopt.geo

from quasr_coil_check import bdistrib_io

PLASMA_PATH = "replicate_lgradb/tmp/wout_surfaces_python_generated.nc"


def bdistrib_for_surfaces(
    plasma_surface: simsopt.geo.SurfaceRZFourier, distance: float, dataset_path: str
):
    cwd = os.path.dirname(PLASMA_PATH)
    netcdf_path = bdistrib_io.write_netcdf(
                    PLASMA_PATH, plasma_surface.to_RZFourier())
    subprocess.check_call(
        [
            "../../bdistrib/bdistrib",
            os.path.basename(bdistrib_io.write_bdistribin(
                os.path.basename(netcdf_path),
                geometry_option=2,
                geometry_info={
                    "separation_outer": distance,
                },
                mpol=12,
                ntor=12,
                dataset_path=dataset_path,
            )),
        ], cwd=cwd
    )

if __name__ == "__main__":
    if os.path.dirname(__file__) == os.getcwd():
        raise RuntimeError(
            "This script should have been excecuted as a module:\npython -m replicate_lgradb.find_single_l"
        )

    for top, dirs, files in os.walk("replicate_lgradb/db"):
        for file in files:
            if os.path.splitext(file)[1] == ".json":
                print(file)
                path = os.path.join(top, file)
                surfs, coils = simsopt.load(path)
                bdistrib_for_surfaces(
                    surfs[-1], 0.5, dataset_path="./replicate_lgradb/tmp/bdistrib_in."+file.removesuffix(".json")
                )
