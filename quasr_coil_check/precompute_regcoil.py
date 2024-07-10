import bdistrib_io
import simsopt
import simsopt.geo
import precompute_surfaces
import subprocess


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
  nlambda = 5
  lambda_min = 1e-30
  lambda_max = 1

  ntheta_plasma = 32
  ntheta_coil   = 33
  nzeta_plasma  = 34
  nzeta_coil    = 35
  mpol_coil = 15
  ntor_coil = 8

  geometry_option_plasma = 2
  wout_filename='{plasma_path}'

  geometry_option_coil=3
  nescin_filename = '{winding_path}'

  net_poloidal_current_Amperes = 1.4
  net_toroidal_current_Amperes = 0.3

  symmetry_option = 3
/
"""
    with open("regcoil_in.python_generated", "w") as f:
        f.write(input_string)
    return subprocess.check_call(["./regcoil/regcoil", "regcoil_in.python_generated"])


if __name__ == "__main__":
    import sys
    import os

    min_ID = 0
    max_ID = 1150000
    if len(sys.argv) == 2:
        max_ID = int(sys.argv[1])
    elif len(sys.argv) >= 3:
        min_ID = int(sys.argv[1])
        max_ID = int(sys.argv[2])
    else:
        print("plase supply a (min and) max ID until which to process the files.")

    for i in range(min_ID, max_ID):
        if os.path.exists(bdistrib_io.get_file_path(i, "simsopt")):
            res = precompute_surfaces.cached_get_surfaces(i)
            run_regcoil(res["lcfs"], res["coil_surf"])
