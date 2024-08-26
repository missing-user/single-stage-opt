import numpy as np
import scipy.optimize
import simsopt
import simsopt.geo
import subprocess
import os
if os.path.dirname(__file__) == os.getcwd():
    raise RuntimeError("This script should have been excecuted as a module:\npython -m replicate_lgradb.find_single_l")
if not "regcoil" in os.getenv("PATH"):
    raise RuntimeError("The regcoil executable should be added to the PATH environment variable:\nexport PATH=\"$PATH:/home/<user>/regcoil\"")
from quasr_coil_check import bdistrib_io
from scipy.io import netcdf_file


REGCOIL_IN_TMP_PATH = "replicate_lgradb/tmp/regcoil_in.python_generated"
REGCOIL_OUT_TMP_PATH = "replicate_lgradb/tmp/regcoil_out.python_generated.nc"
PLASMA_PATH = "replicate_lgradb/tmp/wout_surfaces_python_generated.nc"


def find_regcoil_distance(lcfs):
    """Find the distance the coil surface must be separated from the plasma, to fulfill
    |K|_\infty = 17.16 MA/m
    The optimization is bounded, the surface offset is ]0, 0.5["""
    target_k = 17.16e6

    def kinfty_at_regcoil_distance(l: float):
        # Run Regcoil at offset l
        log = run_regcoil_fixed_dist(lcfs, l).decode("utf-8")
        print(log)
        if "it is too low." in log:
            return np.inf
        if "it is too high." in log:
            return -np.inf
        # Read result, extract K_infty
        with netcdf_file(REGCOIL_OUT_TMP_PATH, "r", mmap=False) as f:
            k_infty = f.variables["max_K"][()]
            return k_infty

    opt_result = scipy.optimize.root_scalar(
        lambda l: kinfty_at_regcoil_distance(float(l)) - target_k, bracket=[1e-3,0.5],# tol=target_k * 1e-6, method="brentq"
    )
    l_result = opt_result.root
    assert l_result > 0
    assert l_result < 0.5
    return l_result


def run_regcoil_fixed_dist(plasma_surface: simsopt.geo.Surface, distance: float):
    bdistrib_io.write_netcdf(PLASMA_PATH, plasma_surface.to_RZFourier())
    cwd = os.path.dirname(REGCOIL_IN_TMP_PATH)

    surface_resolution = 128
    input_string = f"""&regcoil_nml
  general_option = 5 ! Check if target is attainable first
  Nlambda = 16
  ntheta_coil = {surface_resolution}
  ntheta_plasma = {surface_resolution}
  nzeta_coil = {surface_resolution}
  nzeta_plasma = {surface_resolution}
  mpol_potential = 16
  ntor_potential = 16
  target_option = "rms_Bnormal"
  target_value = 0.01 ! Threshold value for LgradB paper

  geometry_option_plasma = 2
  wout_filename='./{PLASMA_PATH.replace(cwd+"/", "")}'

  !geometry_option_coil=3
  !nescin_filename = 'winding_path.replace(cwd+"/", "")'
  geometry_option_coil=2
  separation={distance}

  symmetry_option = 3
/
"""
    with open(REGCOIL_IN_TMP_PATH, "w") as f:
        f.write(input_string)
    print(f"-----------{distance}----------")
    return subprocess.check_output(["regcoil", os.path.basename(REGCOIL_IN_TMP_PATH)], cwd=cwd)
