import numpy as np
import scipy.optimize
import simsopt
import simsopt.geo
import subprocess
import bdistrib_io
from scipy.io import netcdf_file

REGCOIL_IN_TMP_PATH = "tmp/regcoil_in.python_generated"
REGCOIL_OUT_TMP_PATH = "tmp/regcoil_out.python_generated.nc"


def find_regcoil_distance(lcfs):
    """Find the distance the coil surface must be separated from the plasma, to fulfill
    |K|_\infty = 17.16 MA/m
    The optimization is bounded, the surface offset is ]0, 0.5["""
    target_k = 17.16e6

    def kinfty_at_regcoil_distance(l: float):
        # Run Regcoil at offset l
        run_regcoil_fixed_dist(lcfs, l)
        # Read result, extract K_infty
        with netcdf_file(REGCOIL_OUT_TMP_PATH, "r", mmap=False) as f:
            k_infty = f.variables["max_K"][()]
            return k_infty

    opt_result = scipy.optimize.root(
        lambda l: target_k - kinfty_at_regcoil_distance(l), 0.1, tol=target_k * 1e-6
    )
    l_result = opt_result.root
    assert l_result > 0
    assert l_result < 0.5
    return l_result


def run_regcoil_fixed_dist(plasma_surface: simsopt.geo.Surface, distance: float):
    plasma_path = "wout_surfaces_python_generated.nc"
    bdistrib_io.write_netcdf(plasma_path, plasma_surface.to_RZFourier())

    surface_resolution = 128
    input_string = f"""&regcoil_nml
  general_option = 5 ! Check if target is attainable first
  Nlambda = 16
  ntheta_coil = {surface_resolution}
  ntheta_plasma = {surface_resolution}
  nzeta_coil = {surface_resolution}
  nzeta_plasma = {surface_resolution}
  mpol_potential = 18
  ntor_potential = 18
  target_option = "rms_Bnormal"
  target_value = 0.01 ! Threshold value for LgradB paper

  geometry_option_plasma = 2
  wout_filename='./{plasma_path}'

  !geometry_option_coil=3
  !nescin_filename = './winding_path'
  geometry_option_coil=2
  separation={distance}

  symmetry_option = 3
/
"""
    with open(REGCOIL_IN_TMP_PATH, "w") as f:
        f.write(input_string)
    return subprocess.check_call(["../../../regcoil/regcoil", REGCOIL_IN_TMP_PATH])
