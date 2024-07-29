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
        bdistrib_io.write_netcdf(plasma_path, plasma_surface.to_RZFourier())
    elif isinstance(plasma_surface, str):
        plasma_path = plasma_surface

    if isinstance(winding_surface, simsopt.geo.Surface):
        bdistrib_io.write_nescin_file(
            winding_path, winding_surface.to_RZFourier())
    elif isinstance(winding_surface, str):
        winding_path = winding_surface

    input_string = f"""&regcoil_nml
  general_option = 5 ! Check if target is attainable first
  Nlambda = 16
  target_option = "rms_Bnormal"
  target_value = 0.05 ! Arbitrarily chosen based on 3 randomly selected configurations. Will need tweaking

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
    # exit_code = run_regcoil(res["lcfs"], res["middle_surf"])
    if res is None:
        print("Invalid surface for file for ID=", ID)
        return 1
    exit_code = run_regcoil(res["lcfs"], res["coil_surf"])
    if exit_code == 0:
        comppath = bdistrib_io.get_file_path(ID, "regcoil")
        Path(comppath).parent.mkdir(parents=True, exist_ok=True)
        Path(REGCOIL_OUT_TMP_PATH).rename(comppath)
        return exit_code
    return exit_code


def get_regcoil_metrics(ID):
    from scipy.io import netcdf_file
    import numpy as np

    with netcdf_file(bdistrib_io.get_file_path(ID, "regcoil"), "r", mmap=False) as f:
        lambdas = f.variables["lambda"][()]
        regcoil_results = {"lambda": float(lambdas[-1]), "ID": ID}
        print(regcoil_results["lambda"])
        for key in [
            "chi2_B",
            "chi2_K",
            "chi2_Laplace_Beltrami",
            "max_Bnormal",
            "max_K",
        ]:
            metric_for_different_lambda = f.variables[key][()]
            regcoil_results[key] = metric_for_different_lambda
            regcoil_results[key + "[-1]"] = metric_for_different_lambda[-1]
            # regcoil_results[key + " (linear fit)"] = np.polyfit(
            #     metric_for_different_lambda,
            #     lambdas,
            #     1,
            # )[0]
    return regcoil_results


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
