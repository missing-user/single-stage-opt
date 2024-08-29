import numpy as np
import simsopt
import simsopt.geo
import simsopt.field
from scipy.io import netcdf_file
import os
import warnings


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

            if m == 0:
                # Negative mode numbers for m=0 are a weird convention...
                surface.set_rc(m, abs(n), f.variables["rmnc"][-1, i])
                surface.set_zs(m, abs(n), -f.variables["zmns"][-1, i])
            else:
                surface.set_rc(m, n, f.variables["rmnc"][-1, i])
                surface.set_zs(m, n, -f.variables["zmns"][-1, i])

        print("rc10", surface.get_rc(1, 0))
        print("zs10", surface.get_zs(1, 0))
        print("zs21", surface.get_zs(2, 1))
        return surface


def write_netcdf(filename, surface: simsopt.geo.SurfaceRZFourier):
    # HACK: To generate the template file
    # B_VOL_AVG_TARGET = 5.865 #T
    # magnetic_scaling = B_VOL_AVG_TARGET / equil.wout.volavgB

    # from scipy.io import netcdf_file
    # import os

    # filename = "wout_d23p4_tm_reference_LgradBscaling.nc"
    # os.system(f"cp ../../regcoil/equilibria/wout_d23p4_tm.nc {filename}")

    # # Copy the file on disk with a new name, open with r+ and overwrite it.
    # with netcdf_file(filename, "a", mmap=False) as f:
    #     f.variables["bvco"][:] *= magnetic_scaling
    #     f.variables["b0"][()] *= magnetic_scaling
    #     f.variables["bmnc"][:] *= magnetic_scaling
    #     f.variables["volavgB"][()] *= magnetic_scaling
    #     f.variables["bsubsmns"][:] *= magnetic_scaling
    #     f.variables["bsubumnc"][:] *= magnetic_scaling
    #     f.variables["bsubvmnc"][:] *= magnetic_scaling
    #     f.variables["bsupumnc"][:] *= magnetic_scaling
    #     f.variables["bsupvmnc"][:] *= magnetic_scaling

    filename_template = (
        f"{os.path.dirname(__file__)}/wout_d23p4_tm_reference_LgradBscaling.nc"
    )
    os.system(f"cp {filename_template} {filename}")

    # Copy the file on disk with a new name, open with r+ and overwrite it.
    with netcdf_file(filename, "a", mmap=False) as f:
        # print(list(f.variables.keys()))

        # implicitly broadcasts the result throughout all flux surfaces
        mpol = int(f.variables["mpol"][()]) - 1
        ntor = int(f.variables["ntor"][()])
        if mpol + 1 < surface.mpol or ntor < surface.ntor:
            warnings.warn(
                f"Dropping resolution when exporting surface to netcdf_file, due to my terrible, hacky implementation! Original: {surface.mpol, surface.ntor} now: {mpol, ntor}"
            )
        surface.change_resolution(mpol, ntor)
        f.variables["rmnc"][:] = surface.rc.flatten()[surface.ntor:] * 5
        f.variables["zmns"][:] = -surface.zs.flatten()[surface.ntor:] * 5

        # Divided by old nfp multiplied by new nfp
        f.variables["xn"][:] = (
            f.variables["xn"][()] / f.variables["nfp"][()] * surface.nfp
        )
        f.variables["nfp"][()] = surface.nfp
        f.variables["Rmajor_p"][()] = surface.major_radius() * 5
        f.variables["Aminor_p"][()] = surface.minor_radius() * 5

    return filename.replace("wout_", "")


def write_bdistribin(
    netcdf_filename,
    geometry_option=1,
    geometry_info={},
    mpol=12,
    ntor=12,
    dataset_name="python_generated",
):
    filename_out = "bdistrib_in." + dataset_name

    nu = 96
    nv = 96

    transfer_geometry = ""
    if geometry_option == 1:
        if "R0" in geometry_info:
            geometry_info["R0_middle"] = geometry_info["R0"]
            geometry_info["R0_outer"] = geometry_info["R0"]
        transfer_geometry = f"""
    geometry_option_middle=1
    R0_middle = {geometry_info["R0_middle"]}
    a_middle  = {geometry_info["a_middle"]}
    geometry_option_outer=1
    R0_outer = {geometry_info["R0_outer"]}
    a_outer  = {geometry_info["a_outer"]}
        """
    elif geometry_option == 2:
        if "sep_middle" not in geometry_info:
            geometry_info["sep_middle"] = geometry_info["sep_outer"] / 2
        transfer_geometry = f"""
    geometry_option_middle=2
    sep_middle={geometry_info["sep_middle"]}
    geometry_option_outer=2
    sep_outer={geometry_info["sep_outer"]}
        """
    elif geometry_option == 3:
        transfer_geometry = f"""
    geometry_option_middle=3
    nescin_filename_middle='{geometry_info["nescin_filename_middle"]}'
    geometry_option_outer=3
    nescin_filename_outer='{geometry_info["nescin_filename_outer"]}'
        """

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

    {transfer_geometry}

    pseudoinverse_thresholds = 1e-12

    n_singular_vectors_to_save = 16
  /
  """

    with open(filename_out, "w") as f:
        f.write(bdistribin)
    return filename_out


# NESCIN
def read_nescin_file(filename: str, nfp) -> simsopt.geo.SurfaceRZFourier:
    surf = simsopt.geo.SurfaceRZFourier(nfp)
    with open(filename, "r") as f:
        lines = f.readlines()

    mnmax = 0
    for i, line in enumerate(lines):
        if line.startswith("------ Current Surface"):
            mnmax = int(lines[i + 2])
            lines = lines[i + 5:]
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

        surf.set_rc(m, n, float(numbers[2]))
        surf.set_zs(m, n, float(numbers[3]))

    print("rc10", surf.get_rc(1, 0))
    print("zs10", surf.get_zs(1, 0))
    print("zs21", surf.get_zs(2, 1))
    return surf


def write_nescin_file(filename: str, surface: simsopt.geo.SurfaceRZFourier):
    with open(filename, "w") as f:
        f.write(f"\n------ Current Surface: Coil-Plasma separation = 0.0 ----\n")
        f.write("Number of fourier modes in table\n")
        num_modes = len(surface.m) // 2 + 1
        f.write(f" {num_modes}\n")

        f.write("Table of fourier coefficients\n")
        f.write("m,n,crc,czs,crs,czc\n")

        for i in range(num_modes):
            m = surface.m[i]
            n = surface.n[i]

            f.write(
                f" {m} {n:+2d} {surface.get_rc(m, n): .12E} {surface.get_zs(m, n): .12E} {surface.get_rs(m, n) if not surface.stellsym else 0: .12E} {surface.get_zc(m, n)  if not surface.stellsym else 0: .12E}\n"
            )


def get_file_path(ID, file_type="simsopt"):
    """Returns the relative file paths into the QUASR db folder structure for one of then following file file_types:
    [simsopt, complexity, nml, bdistrib, regcoil, surfaces]"""
    fID = ID // 1000
    if file_type == "simsopt":
        return f"{os.path.dirname(__file__)}/QUASR_db/simsopt_serials/{fID:04}/serial{ID:07}.json"
    elif file_type == "complexity":
        return f"{os.path.dirname(__file__)}/QUASR_db/complexities/{fID:04}/complexity{ID:07}.json"
    elif file_type == "nml":
        return f"{os.path.dirname(__file__)}/QUASR_db/nml/{fID:04}/input{ID:07}"
    elif file_type == "bdistrib":
        return f"{os.path.dirname(__file__)}/QUASR_db/bdistrib_serials/{fID:04}/bdistrib_out.{ID:07}.nc"
    elif file_type == "regcoil":
        return f"{os.path.dirname(__file__)}/QUASR_db/regcoil/{fID:04}/regcoil_out.{ID:07}.nc"
    elif file_type == "surfaces":
        return f"{os.path.dirname(__file__)}/QUASR_db/surfaces/{fID:04}/surfaces{ID:07}.json"
    else:
        raise RuntimeError()


def load_simsopt_up_to(max_ID):
    objs = []
    for i in range(max_ID):
        sopt_path = get_file_path(i, file_type="simsopt")
        if os.path.exists(get_file_path(i, file_type="bdistrib")):
            try:
                obj = simsopt.load(sopt_path)
                objs.append({"surfaces": obj[0], "coils": obj[1], "ID": i})
            except:
                pass
    return objs


# UNIT TESTS

if __name__ == "__main__":
    import random

    random.seed(42)
    import os

    def compare_surfaces(s1, s2):
        assert s1.ntor == s2.ntor
        assert s1.mpol == s2.mpol
        assert s1.nfp == s2.nfp
        assert np.allclose(s1.zs, s2.zs)
        assert np.allclose(s1.rc, s2.rc)
        assert np.allclose(s1.zc, s2.zc)
        assert np.allclose(s1.rs, s2.rs)
        return True

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
    os.remove("unit_test.nc")
    # os.remove("nescin.unit_test")

    print("Success")
