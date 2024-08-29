import subprocess
import simsopt
import simsopt.geo

from quasr_coil_check import bdistrib_io

PLASMA_PATH = "replicate_lgradb/tmp/wout_surfaces_python_generated.nc"


def bdistrib_for_surfaces(
    plasma_surface: simsopt.geo.SurfaceRZFourier, distance: float, dataset_name: str
):
    subprocess.check_call(
        [
            "../bdistrib/bdistrib",
            bdistrib_io.write_bdistribin(
                bdistrib_io.write_netcdf(
                    PLASMA_PATH, plasma_surface.to_RZFourier()),
                geometry_option=2,
                geometry_info={
                    "sep_outer": distance,
                },
                mpol=16,
                ntor=16,
                dataset_name=dataset_name,
            ),
        ]
    )
    subprocess.call(
        ["rm", "*.dat"]
    )  # Delete Debug Logs cause I don't know how to disable them


if __name__ == "__main__":
    import os

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
                    surfs[-1], 0.1, dataset_name=file.removesuffix(".json")
                )
