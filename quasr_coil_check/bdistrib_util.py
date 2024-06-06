import numpy as np
from scipy.spatial.distance import cdist
import simsopt
import simsopt.geo
import simsopt.field
from scipy.io import netcdf_file
import os


def lhs_to_rhs_surface(surface: simsopt.geo.SurfaceRZFourier):
    # Right and left handed coordinates. Flip the signs of sin(theta) to match
    surface.zs = -surface.zs
    surface.rs = -surface.rs
    return surface


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


def read_netcdf(filename: str):
    with netcdf_file(filename, "r", mmap=False) as f:
        mpol = int(f.variables["mpol"][()]) - 1
        ntor = int(f.variables["ntor"][()])
        nfp = int(f.variables["nfp"][()])
        surface = simsopt.geo.SurfaceRZFourier(nfp)
        surface.change_resolution(mpol, ntor)

        mnmax = f.variables["xm"][()].shape[0]

        for i, m, nnfp in zip(
            range(mnmax), f.variables["xm"][()], f.variables["xn"][()]
        ):
            m = int(m)
            n = int(nnfp / f.variables["nfp"][()])
            surface.set_rc(m, n, f.variables["rmnc"][-1, i])
            surface.set_zs(m, n, -f.variables["zmns"][-1, i])

        return surface


def write_netcdf(filename, surface: simsopt.geo.SurfaceRZFourier):
    filename_template = "../bdistrib/equilibria/wout_w7x_standardConfig.nc"
    os.system(f"cp {filename_template} {filename}")

    # Copy the file on disk with a new name, open with r+ and overwrite it.
    with netcdf_file(filename, "a", mmap=False) as f:
        # print(list(f.variables.keys()))

        # implicitly broadcasts the result throughout all flux surfaces
        mpol = int(f.variables["mpol"][()]) - 1
        ntor = int(f.variables["ntor"][()])
        surface.change_resolution(mpol, ntor)
        f.variables["rmnc"][:] = surface.rc.flatten()[surface.ntor :]
        f.variables["zmns"][:] = -surface.zs.flatten()[surface.ntor :]

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

    return filename.replace("wout_", "")


def write_bdistribin(
    netcdf_filename,
    mpol=8,
    ntor=8,
    sep_outer=0.1,
    middle_surface_filename=None,
    outer_surface_filename=None,
):
    nu = 64
    nv = 128

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
    wout_filename='{netcdf_filename}'

    geometry_option_middle={2 if middle_surface_filename is None else 3}
    separation_middle={sep_middle}
    nescin_filename_middle='{middle_surface_filename}'

    geometry_option_outer={2 if outer_surface_filename is None else 3}
    separation_outer={sep_outer}
    nescin_filename_outer='{outer_surface_filename}'

    pseudoinverse_thresholds = 1e-12

    n_singular_vectors_to_save = 16
  /
  """
    filename_out = "bdistrib_in.python_generated"
    with open(filename_out, "w") as f:
        f.write(bdistribin)
    return filename_out


############# NESCIN


def read_nescin_file(filename: str, nfp):
    surf = simsopt.geo.SurfaceRZFourier(nfp)
    with open(filename, "r") as f:
        lines = f.readlines()

    mnmax = 0
    for i, line in enumerate(lines):
        if line.startswith("------ Current Surface"):
            mnmax = int(lines[i + 2])
            lines = lines[i + 5 :]
            break
    assert len(lines) == mnmax

    mmax = 0
    nmax = 0
    for line in lines:
        numbers = line.split()
        m = int(numbers[0])
        n = int(numbers[1])
        mmax = max(abs(m), mmax)
        nmax = max(abs(n), nmax)
    surf.change_resolution(mmax, nmax)

    for line in lines:
        numbers = line.split()
        m = int(numbers[0])
        n = int(numbers[1])
        if m == 0:
            n = abs(n)
        surf.set_rc(m, n, float(numbers[2]))
        surf.set_zs(m, n, float(numbers[3]))

    return surf


def write_nescin_file(filename: str, surface: simsopt.geo.SurfaceRZFourier):
    with open(filename, "w") as f:
        f.write(f"\n------ Current Surface: Coil-Plasma separation = 0.0 ----\n")
        f.write("Number of fourier modes in table\n")
        num_modes = len(surface.m) // 2 + 1
        f.write(f" {num_modes}\n")

        f.write("Table of fourier coefficients\n")
        f.write("m,n,crc,czs,crs,czc\n")
        m = surface.m
        n = surface.n

        for i in range(num_modes):
            f.write(
                f" {m[i]} {n[i]:+2d} {surface.get_rc(m[i], n[i]): .12E} {surface.get_zs(m[i], n[i]): .12E} {surface.get_rs(m[i], n[i]) if not surface.stellsym else 0: .12E} {surface.get_zc(m[i], n[i])  if not surface.stellsym else 0: .12E}\n"
            )


##### UNIT TESTS

if __name__ == "__main__":
    import random

    def compare_surfaces(s1, s2):
        assert s1.ntor == s2.ntor
        assert s1.mpol == s2.mpol
        return (
            s1.nfp == s2.nfp
            and s1.ntor == s2.ntor
            and s1.mpol == s2.mpol
            and np.allclose(s1.rc, s2.rc)
            and np.allclose(s1.zs, s2.zs)
            and np.allclose(s1.zc, s2.zc)
            and np.allclose(s1.rs, s2.rs)
        )

    surf = simsopt.geo.SurfaceRZFourier(5, ntor=5, mpol=3)
    assert compare_surfaces(surf, surf)  # Confirm comparison works at all

    for m in range(3 + 1):
        nmin = -5 if m > 0 else 0
        for n in range(nmin, 5 + 1):
            surf.set_rc(m, n, random.random())
            surf.set_zs(m, n, random.random())

    print("Nescin")
    write_nescin_file("nescin.unit_test", surf)
    nescinsurf = read_nescin_file("nescin.unit_test", 5)
    assert compare_surfaces(surf, nescinsurf)

    print("NetCDF")
    write_netcdf("unit_test.nc", surf)
    ncdfsurf2 = read_netcdf("unit_test.nc")
    assert compare_surfaces(surf, ncdfsurf2)

    # lhs is an inplace operation, does not return a copy
    assert not compare_surfaces(surf, lhs_to_rhs_surface(ncdfsurf2))
    assert compare_surfaces(surf, lhs_to_rhs_surface(ncdfsurf2))

    print("Success")
