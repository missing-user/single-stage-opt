import bdistrib_io
import numpy as np
from simsopt.objectives import QuadraticPenalty
import simsopt
import simsopt.geo
import os
import json

from skimage.filters import window

from scipy.spatial.distance import cdist
from pathlib import Path
import precompute_surfaces


def coil_surf_distance(curves, lcfs) -> np.ndarray:
    pointcloud1 = lcfs.gamma().reshape((-1, 3))
    distances = [np.min(cdist(pointcloud1, c.gamma()), axis=0) for c in curves]
    return np.array(distances).T


def compute_coil_surf_dist(coils, lcfs):
    curves = [c.curve for c in coils]
    return coil_surf_distance(curves, lcfs)


def compute_complexity(ID):
    surfaces, coils = simsopt.load(bdistrib_io.get_file_path(ID, "simsopt"))
    lcfs: simsopt.geo.SurfaceRZFourier = surfaces[-1]
    curves = [c.curve for c in coils]

    # Form the total objective function.
    LENGTH_WEIGHT = 1
    TARGET_LENGTH = 1.0 * lcfs.minor_radius() * 2 * np.pi
    CC_WEIGHT = 1.0
    CC_THRESHOLD = 0.1
    CURVATURE_THRESHOLD = 0
    CURVATURE_WEIGHT = 1

    MSC_THRESHOLD = 0
    MSC_WEIGHT = 1

    Jls = LENGTH_WEIGHT * sum(
        [QuadraticPenalty(simsopt.geo.CurveLength(c), TARGET_LENGTH)
         for c in curves]
    )
    Jccdist = CC_WEIGHT * simsopt.geo.CurveCurveDistance(curves, CC_THRESHOLD)
    Jcsdist = simsopt.geo.CurveSurfaceDistance(
        curves, lcfs, lcfs.minor_radius())
    Jcs = CURVATURE_WEIGHT * sum(
        [simsopt.geo.LpCurveCurvature(c, 2, 0.5) for c in curves]
    )
    Jmscs = MSC_WEIGHT * sum(
        [
            QuadraticPenalty(simsopt.geo.MeanSquaredCurvature(
                c), MSC_THRESHOLD, "max")
            for c in curves
        ]
    )

    JF = Jccdist + Jcsdist + Jcs + Jmscs  # +Jls +
    return {
        "complexity": float(JF.J()),  # type: ignore
        "Jls": float(Jls.J()),  # type: ignore
        "Jccdist": float(Jccdist.J()),  # type: ignore
        # type: ignore
        "coil surface dist": float(np.min(compute_coil_surf_dist(coils, lcfs))),
        "Jcs": float(Jcs.J()),  # type: ignore
        "Jmscs": float(Jmscs.J()),  # type: ignore
        "nfp": int(lcfs.nfp),
        "n_coils": len(coils),
    }


def compute_and_store_complexity(ID):
    complexity = compute_complexity(ID)
    comppath = bdistrib_io.get_file_path(ID, "complexity")
    Path(comppath).parent.mkdir(parents=True, exist_ok=True)
    with open(comppath, "w") as f:
        json.dump(complexity, f)
    return complexity


def spectral_power(Bn: np.ndarray):
    Bn = np.array(Bn)
    w = 1.0 - window(25, (min(Bn.shape), min(Bn.shape)))
    center_x = (Bn.shape[0] - w.shape[0]) // 2
    center_y = (Bn.shape[1] - w.shape[1]) // 2
    w_padded = np.ones_like(Bn)
    w_padded[center_x: center_x + w.shape[0],
             center_y: center_y + w.shape[1]] = w
    fftImg = np.fft.fft2(Bn)
    windowedImg = np.fft.fftshift(w_padded) * np.abs(fftImg)

    return np.mean(windowedImg)


def possibly_add_spectral_power(complexity_dict: dict, ID, comppath):
    if "spectral_power" not in complexity_dict:
        j_surfaces = precompute_surfaces.cached_get_surfaces(ID)
        if j_surfaces is None:
            print(ID, "has no associated surfaces")
            return complexity_dict

        complexity_dict["spectral_power"] = spectral_power(j_surfaces["BdotN"])
        with open(comppath, "w") as f:
            json.dump(complexity_dict, f)

    return complexity_dict


def cached_get_complexity(ID):
    comppath = bdistrib_io.get_file_path(ID, "complexity")
    if Path(comppath).exists():
        with open(bdistrib_io.get_file_path(ID, "complexity")) as f:
            j_complexity = json.load(f)
        # possibly_add_spectral_power(j_complexity, ID, comppath)
    else:
        j_complexity = compute_and_store_complexity(ID)
        possibly_add_spectral_power(j_complexity, ID, comppath)
    j_complexity["ID"] = ID
    return j_complexity


if __name__ == "__main__":
    import sys

    min_ID = 0
    max_ID = 0
    if len(sys.argv) == 2:
        max_ID = int(sys.argv[1])
    elif len(sys.argv) == 3:
        min_ID = int(sys.argv[1])
        max_ID = int(sys.argv[2])
    else:
        print("plase supply a (min and) max ID until which to process the files.")

    print("Computing complexities up to ID", max_ID)
    for i in range(min_ID, max_ID):
        if os.path.exists(bdistrib_io.get_file_path(i, "simsopt")):
            print(i)
            cached_get_complexity(i)
