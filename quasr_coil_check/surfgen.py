import numpy as np
import simsopt.geo
import simsopt.geo.jit
import scipy.optimize

from jax import grad

# from scipy.spatial.distance import cdist
import jax.numpy as jnp
from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec
from simsopt.solve import least_squares_serial_solve, serial_solve
import simsopt.objectives
import simsopt.solve


@simsopt.geo.jit.jit
def cdist(xyz1, xyz2):
    dists = jnp.linalg.norm(xyz1[:, np.newaxis, :] - xyz2[np.newaxis, :, :], axis=-1)
    # dists1 = np.array([np.linalg.norm(xyz1 - point, axis=-1) for point in xyz2])
    # assert np.allclose(dists1, dists)
    return dists


@simsopt.geo.jit.jit
def min_cdist(xyz1, xyz2):
    return jnp.min(
        cdist(xyz1, xyz2),
        axis=0,
    )


class SurfaceSurfaceDistance(Optimizable):
    def __init__(
        self,
        static_surface: simsopt.geo.SurfaceRZFourier,
        moving_surface: simsopt.geo.SurfaceRZFourier,
    ):
        self.moving_surface = moving_surface

        # Partial argument application for this static surface
        def my_min_cdist(xyz):
            # if target_distance is None:
            return min_cdist(static_surface.gamma().reshape((-1, 3)), xyz)
            # else:
            #     return (
            #         jnp.sum(
            #             min_cdist(static_surface.gamma().reshape((-1, 3)), xyz)
            #             - target_distance
            #         )
            #         ** 2
            #     )

        self.myJ = simsopt.geo.jit.jit(my_min_cdist)
        self.myGradJ = simsopt.geo.jit.jit(lambda xyz: grad(my_min_cdist)(xyz))

        Optimizable.__init__(self, depends_on=[moving_surface])

    def J(self):
        # self.moving_surface.plot()
        return self.myJ(
            self.moving_surface.gamma().reshape((-1, 3)),
        )

    return_fn_map = {"J": J}


def surfgen(
    surface: simsopt.geo.SurfaceRZFourier,
    target_distance: float,
    iterative_constraits=False,
):
    surface_copy = simsopt.geo.SurfaceRZFourier.from_nphi_ntheta(
        32, 32, nfp=surface.nfp
    )
    surface_copy.change_resolution(surface.mpol, surface.ntor)
    surface_copy.rc = surface.rc
    surface_copy.zs = surface.zs

    wrapping_surf = simsopt.geo.SurfaceRZFourier.from_nphi_ntheta(
        32, 32, nfp=surface_copy.nfp
    )
    wrapping_surf.change_resolution(1, 1)

    for m, n in zip(wrapping_surf.m, wrapping_surf.n):
        wrapping_surf.set_rc(m, n, surface_copy.get_rc(m, n))
        wrapping_surf.set_zs(m, n, surface_copy.get_zs(m, n))

    wrapping_surf.change_resolution(1, 1)
    wrapping_surf.scale(1.5)
    wrapping_surf.fix("rc(0,0)")
    wrapping_surf.set_lower_bound("rc(1,0)", wrapping_surf.get_rc(1, 0))
    wrapping_surf.set_lower_bound("zs(1,0)", wrapping_surf.get_zs(1, 0))

    surf_surf_dist = SurfaceSurfaceDistance(
        surface_copy,
        wrapping_surf,  # target_distance=target_distance
    )

    problem = simsopt.objectives.LeastSquaresProblem(
        np.array([target_distance]), np.array([1.0]), [surf_surf_dist.J]
    )

    if iterative_constraits:
        for fourier_resolution in range(3):
            least_squares_serial_solve(problem, ftol=1e-5)

            wrapping_surf.change_resolution(
                wrapping_surf.mpol + 1, wrapping_surf.ntor + 1
            )
            wrapping_surf.fixed_range(
                0, fourier_resolution, fourier_resolution, fourier_resolution
            )
    else:
        wrapping_surf.change_resolution(3, 2)
        least_squares_serial_solve(problem, ftol=1e-5)

    return wrapping_surf


if __name__ == "__main__":
    simple_torus = simsopt.geo.SurfaceRZFourier(5)

    surfgen(simple_torus, 0.2).plot()
