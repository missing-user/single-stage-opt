import numpy as np
from scipy.spatial.distance import cdist
import simsopt
import simsopt.geo
import simsopt.field
from scipy.io import netcdf_file
import os


def netcdf_from_surface(surface: simsopt.geo.SurfaceRZFourier):

    filename_out = "wout_surfaces_python_generated.nc"
    filename = "../bdistrib/equilibria/wout_w7x_standardConfig.nc"
    os.system(f"cp {filename} {filename_out}")

    # Copy the file on disk with a new name, open with r+ and overwrite it.
    with netcdf_file(filename_out, "a", mmap=False) as f:
        print(list(f.variables.keys()))
        print(
            f.variables["zmns"][()].shape,
            f.variables["zmns"].units,
            f.variables["zmns"].dimensions,
            f.variables["zmns"][()].dtype,
        )
        print(np.max(f.variables["zmns"][()]))

        # The plasma surface read in by bdistrib is zmns[ns] & rmnc[ns]
        # nfp_vmec = nfp
        # Rmajor_p = R0

        # implicitly broadcasts the result throughout all flux surfaces
        mpol = int(f.variables["mpol"][()]) - 1
        ntor = int(f.variables["ntor"][()])
        surface.change_resolution(mpol, ntor)
        f.variables["rmnc"][:] = surface.rc.flatten()[surface.ntor :]
        f.variables["zmns"][:] = -surface.zs.flatten()[
            surface.ntor :
        ]  # Right and left handed coordinates. Flip the signs of sin(theta) to match

        f.variables["Rmajor_p"][()] = surface.major_radius()
        f.variables["nfp"][()] = surface.nfp

        # TODO: Net poloidal current profile (bvco), net poloidal current Amperes is computed from this
        # net_poloidal_current_Amperes = (2*pi/mu0) * (1.5*bvco(end) - 0.5*bvco(end-1));
        # f.variables["bvco"][:] = np.zeros(f.variables["lmns"][()].shape)

        # TODO: set lmns component? sinmn component of lambda, half mesh
        # TODO: GMNC component? Both dont seem to have an impact
        # f.variables["gmnc"][:] = np.zeros(f.variables["gmnc"][()].shape)
        # f.variables["lmns"][:] = np.zeros(f.variables["lmns"][()].shape)

        # HACK sets the success flag to true so the input reading doesnt fail
        f.variables["ier_flag"][()] = 0

    return filename_out.replace("wout_", "")


def write_bdistribin(netcdffilename, mpol=14, ntor=14, sep_outer=0.1):
    nu = 64
    nv = 256

    sep_middle = sep_outer / 2

    bdistribin = f"""&bdistrib
    transfer_matrix_option = 1

    nu_plasma={nu}
    nu_middle={nu}
    nu_outer ={nu}

    nv_plasma={nv}
    nv_middle={nv}
    nv_outer ={nv}

    ! This run is not well resolved at this resolution, but it is enough for testing.
    mpol_plasma = {mpol}
    mpol_middle = {mpol}
    mpol_outer  = {mpol}

    ntor_plasma = {ntor}
    ntor_middle = {ntor}
    ntor_outer  = {ntor}

    geometry_option_plasma = 2
    wout_filename='{netcdffilename}'

    geometry_option_middle=2
    separation_middle={sep_middle}

    geometry_option_outer=2
    separation_outer={sep_outer}

    pseudoinverse_thresholds = 1e-12

    n_singular_vectors_to_save = 16
  /
  """
    filename_out = "bdistrib_in.python_generated"
    with open(filename_out, "w") as f:
        f.write(bdistribin)
    return filename_out


def minimum_coil_surf_distance(curves, lcfs) -> float:
    min_dist = np.inf
    pointcloud1 = lcfs.gamma().reshape((-1, 3))
    for c in curves:
        pointcloud2 = c.gamma()
        min_dist = min(min_dist, np.min(cdist(pointcloud1, pointcloud2)))
        # Equivalent python code:
        # for point in pointcloud2:
        #   min_dist = min(min_dist, np.min(np.linalg.norm(pointcloud1 - point, axis=-1)))

    return float(min_dist)
