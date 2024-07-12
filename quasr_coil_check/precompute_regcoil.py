import bdistrib_io
import simsopt
import simsopt.geo
import precompute_surfaces
import subprocess
from pathlib import Path

REGCOIL_IN_TMP_PATH = "regcoil_in.python_generated"
REGCOIL_OUT_TMP_PATH = "regcoil_out.python_generated.nc"


def run_regcoil(
    plasma_surface: simsopt.geo.SurfaceRZFourier | str,
    winding_surface: simsopt.geo.SurfaceRZFourier | str,
):
    plasma_path = "wout_surfaces_python_generated.nc"
    winding_path = "nescin.winding_surface"
    if isinstance(plasma_surface, simsopt.geo.Surface):
        bdistrib_io.write_netcdf(
            "wout_surfaces_python_generated.nc", plasma_surface.to_RZFourier()
        )
    elif isinstance(plasma_surface, str):
        plasma_path = plasma_surface

    if isinstance(winding_surface, simsopt.geo.Surface):
        bdistrib_io.write_nescin_file(
            "nescin.winding_surface", winding_surface.to_RZFourier()
        )
    elif isinstance(winding_surface, str):
        winding_path = winding_surface

    input_string = f"""&regcoil_nml
  general_option = 1
  ! nlambda = 20

  geometry_option_plasma = 2
  wout_filename='./{plasma_path}'

  geometry_option_coil=3
  nescin_filename = './{winding_path}'

  symmetry_option = 3
/
"""
    with open(REGCOIL_IN_TMP_PATH, "w") as f:
        f.write(input_string)
    return subprocess.check_call(["../regcoil/regcoil", REGCOIL_IN_TMP_PATH])


def compute_and_save(ID: int):
    res = precompute_surfaces.cached_get_surfaces(ID)
    exit_code = run_regcoil(res["lcfs"], res["middle_surf"])
    # exit_code = run_regcoil(res["lcfs"], res["coil_surf"])
    if exit_code == 0:
        comppath = bdistrib_io.get_file_path(ID, "regcoil")
        Path(comppath).parent.mkdir(parents=True, exist_ok=True)
        Path(REGCOIL_OUT_TMP_PATH).rename(comppath)


if __name__ == "__main__":
    import sys

    min_ID = 0
    max_ID = 1150000
    if len(sys.argv) == 2:
        max_ID = int(sys.argv[1])
    elif len(sys.argv) >= 3:
        min_ID = int(sys.argv[1])
        max_ID = int(sys.argv[2])
    else:
        print("Defaulting to range ", min_ID, max_ID)

    for i in range(min_ID, max_ID):
        if Path(bdistrib_io.get_file_path(i, "simsopt")).exists():
            compute_and_save(i)
