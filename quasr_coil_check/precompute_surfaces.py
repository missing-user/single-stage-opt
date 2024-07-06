import bdistrib_io
import numpy as np
import simsopt.field
import simsopt
import simsopt.geo
import os
import json

from pathlib import Path
import surfgen


def compute_B(coils, surface) -> np.ndarray:
    bs = simsopt.field.BiotSavart(coils)
    bs.set_points(surface.gamma().reshape((-1, 3)))
    return bs.B()


def compute_Bn(B, surface) -> np.ndarray:
    normal = surface.unitnormal()
    return np.sum(B.reshape(normal.shape) * normal, axis=2)[:, :, None]


def compute_surfaces(ID):
    surfaces, coils = simsopt.load(bdistrib_io.get_file_path(ID, "simsopt"))
    lcfs = surfaces[-1].to_RZFourier()

    plasma_coil_distances = surfgen.coil_to_surface_distances(coils, lcfs)
    # assert (plasma_coil_distances > 0).all(), "the coils cannot be inside the plasma"
    if (plasma_coil_distances > 0).all():
        print(ID, "the coils cannot be inside the plasma")
        return None
    target_distance = min(plasma_coil_distances)

    middle_surf = surfgen.surfgen(lcfs, target_distance * 0.5)
    coil_surf = surfgen.surfgen(lcfs, target_distance, initial_guess=middle_surf)
    # simsopt.geo.plot([middle_surf, coil_surf]+coils, engine="plotly", show=False, close=True)

    B = compute_B(coils, middle_surf)
    BdotN = compute_Bn(B, middle_surf)

    return {
        "lcfs": lcfs,
        "middle_surf": middle_surf,
        "coil_surf": coil_surf,
        "B": B,
        "BdotN": BdotN,
    }


def compute_and_store_surfaces(ID):
    surfaces = compute_surfaces(ID)
    comppath = bdistrib_io.get_file_path(ID, "surfaces")
    Path(comppath).parent.mkdir(parents=True, exist_ok=True)
    simsopt.save(surfaces, comppath)
    return surfaces


def cached_get_surfaces(ID):
    comppath = bdistrib_io.get_file_path(ID, "surfaces")
    if os.path.exists(comppath):
        return simsopt.load(bdistrib_io.get_file_path(ID, "surfaces"))
    else:
        return compute_and_store_surfaces(ID)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        min_ID = 0
        max_ID = int(sys.argv[1])
    elif len(sys.argv) == 3:
        min_ID = int(sys.argv[1])
        max_ID = int(sys.argv[2])
    else:
        print("plase supply a (min and) max ID until which to process the files.")

    print("Computing surfaces up to ID", max_ID)
    for i in range(min_ID, max_ID):
        if os.path.exists(bdistrib_io.get_file_path(i, "simsopt")):
            print(i)
            cached_get_surfaces(i)
