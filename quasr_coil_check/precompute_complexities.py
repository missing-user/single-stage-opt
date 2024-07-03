import bdistrib_io
import numpy as np
from simsopt.objectives import QuadraticPenalty
import simsopt
import simsopt.geo
import os
import json

from pathlib import Path


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
        [QuadraticPenalty(simsopt.geo.CurveLength(c), TARGET_LENGTH) for c in curves]
    )
    Jccdist = CC_WEIGHT * simsopt.geo.CurveCurveDistance(curves, CC_THRESHOLD)
    Jcsdist = simsopt.geo.CurveSurfaceDistance(curves, lcfs, lcfs.minor_radius())
    Jcs = CURVATURE_WEIGHT * sum(
        [simsopt.geo.LpCurveCurvature(c, 2, 0.5) for c in curves]
    )
    Jmscs = MSC_WEIGHT * sum(
        [
            QuadraticPenalty(simsopt.geo.MeanSquaredCurvature(c), MSC_THRESHOLD, "max")
            for c in curves
        ]
    )

    JF = Jccdist + Jcsdist + Jcs + Jmscs  # +Jls +
    return {
        "complexity": float(JF.J()),  # type: ignore
        "Jls": float(Jls.J()),  # type: ignore
        "Jccdist": float(Jccdist.J()),  # type: ignore
        "Jcs": float(Jcs.J()),  # type: ignore
        "Jmscs": float(Jmscs.J()),  # type: ignore
        "nfp": int(lcfs.nfp),
        "volume": int(lcfs.volume()),
        "n_coils": len(coils),
    }


def compute_and_store_complexity(ID):
    complexity = compute_complexity(ID)
    comppath = bdistrib_io.get_file_path(ID, "complexity")
    Path(comppath).parent.mkdir(parents=True, exist_ok=True)
    with open(comppath, "w") as f:
        json.dump(complexity, f)
    return complexity



def cached_get_complexity(ID):
    comppath = bdistrib_io.get_file_path(ID, "complexity")
    if os.path.exists(comppath):
        with open(bdistrib_io.get_file_path(ID, "complexity")) as f:
            return json.load(f)
    else:
        return compute_and_store_complexity(ID)


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

    print("Computing complexities up to ID", max_ID)
    for i in range(min_ID, max_ID):
        if os.path.exists(bdistrib_io.get_file_path(i, "simsopt")):
            print(i)
            cached_get_complexity(i)
