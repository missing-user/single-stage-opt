import simsopt.geo
import simsopt.geo.jit
import scipy.optimize

from jax import grad

# from scipy.spatial.distance import cdist
import jax.numpy as jnp
from simsopt._core import Optimizable
import simsopt.objectives
import simsopt.solve


class SurfaceSurfaceDistance(Optimizable):
    def __init__(
        self,
        static_surface: simsopt.geo.SurfaceRZFourier,
        moving_surface: simsopt.geo.SurfaceRZFourier,
        target_distance: float = 0.0,
    ):
        self.moving_surface = moving_surface

        # Partial argument application for this static surface
        def my_min_cdist(xyz):
            return (
                -simsopt.geo.signed_distance_from_surface(xyz, static_surface)
                - target_distance
            )

        self.myJ = my_min_cdist
        self.myGradJ = simsopt.geo.jit.jit(lambda xyz: grad(my_min_cdist)(xyz))

        Optimizable.__init__(self, depends_on=[moving_surface])

    def J(self):
        return self.myJ(
            self.moving_surface.gamma().reshape((-1, 3)),
        )

    return_fn_map = {"J": J}


def deep_copy_surf(surface: simsopt.geo.SurfaceRZFourier):
    surface_copy = simsopt.geo.SurfaceRZFourier.from_nphi_ntheta(
        32, 32, nfp=surface.nfp
    )
    surface_copy.change_resolution(surface.mpol, surface.ntor)
    surface_copy.rc = surface.rc
    surface_copy.zs = surface.zs
    return surface_copy


def surfgen(
    surface: simsopt.geo.SurfaceRZFourier,
    target_distance: float,
    iterative_constraits=0,
):
    # Needs to be copied to get the full torus representation,
    # even if the underlying surface was just a half period
    surface_copy = deep_copy_surf(surface)
    wrapping_surf = deep_copy_surf(surface)

    wrapping_surf.scale(1.5)
    wrapping_surf.fix("rc(0,0)")
    wrapping_surf.set_lower_bound("rc(1,0)", wrapping_surf.get_rc(1, 0))
    wrapping_surf.set_lower_bound("zs(1,0)", wrapping_surf.get_zs(1, 0))
    wrapping_surf.change_resolution(1, 1)

    surf_surf_dist = SurfaceSurfaceDistance(
        surface_copy,
        wrapping_surf,  # target_distance=target_distance
    )
    problem = simsopt.objectives.LeastSquaresProblem(
        jnp.array([target_distance]), jnp.array([1.0]), [surf_surf_dist.J]
    )

    if not iterative_constraits or iterative_constraits == 0:
        wrapping_surf.change_resolution(3, 2)
        iterative_constraits = 1
    for fourier_resolution in range(iterative_constraits):
        # The simsopt solve prints incredible amounts of log messages. simsopt.util.log(0) does not fix it.
        # result = simsopt.solve.least_squares_serial_solve(
        #     problem, ftol=1e-5, max_nfev=50
        # )
        result = scipy.optimize.least_squares(
            problem.residuals, problem.x.copy(), ftol=1e-5, max_nfev=50
        )
        problem.x = result.x

        wrapping_surf.change_resolution(wrapping_surf.mpol + 1, wrapping_surf.ntor + 1)
        wrapping_surf.fixed_range(
            0, fourier_resolution, fourier_resolution, fourier_resolution
        )

    return wrapping_surf


def surface_between_plasma_coils(
    curves: list[simsopt.geo.CurveRZFourier],
    surface: simsopt.geo.SurfaceRZFourier,
    fraction: float = 0.5,
):
    """Compute a SurfaceRZFourier that is placed between the plasma surface and the coils.
    If fraction=0 it is exactly on the plasma surface, fraction=1 should become the coil winding surface.
    """
    surf2 = deep_copy_surf(surface)
    wrapping_surf = deep_copy_surf(surface)
    wrapping_surf.change_resolution(3, 2)
    wrapping_surf.set_lower_bound("rc(0,0)", wrapping_surf.get_rc(0, 0))
    wrapping_surf.set_lower_bound("rc(1,0)", wrapping_surf.get_rc(1, 0))
    wrapping_surf.set_lower_bound("zs(1,0)", wrapping_surf.get_zs(1, 0))
    surf2.fix_all()

    for c in curves:
        c.fix_all()
    cpoints = jnp.array([c.gamma() for c in curves]).reshape((-1, 3))

    plasma_coil_distances = -simsopt.geo.signed_distance_from_surface(cpoints, surf2)
    target_distance = min(plasma_coil_distances) * fraction
    print("target_distance: ", target_distance)

    # For minimum distance away from both surfaces
    cs = simsopt.geo.CurveSurfaceDistance(curves, surf2, target_distance)
    ss = SurfaceSurfaceDistance(surf2, wrapping_surf, target_distance)

    problem = simsopt.objectives.LeastSquaresProblem(
        jnp.array([0, 0]), jnp.array([1, 1]), [cs.J, ss.J]
    )

    result = scipy.optimize.least_squares(
        problem.residuals, problem.x.copy(), ftol=1e-5, max_nfev=50
    )
    problem.x = result.x
    return wrapping_surf


if __name__ == "__main__":
    simple_torus = simsopt.geo.SurfaceRZFourier(5)

    surfgen(simple_torus, 0.1).plot()
    surfgen(simple_torus, 0.2, iterative_constraits=3).plot()
    surfgen(simple_torus, 0.3).plot()
