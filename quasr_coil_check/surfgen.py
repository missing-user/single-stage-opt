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
        min_distance: float = 0.0,
    ):
        self.moving_surface = moving_surface

        # Partial argument application for this static surface
        def my_min_cdist(xyz):
            return jnp.maximum(
                0,
                -simsopt.geo.signed_distance_from_surface(xyz, static_surface)
                - min_distance,
            )

        self.myJ = my_min_cdist
        self.myGradJ = simsopt.geo.jit.jit(lambda xyz: grad(my_min_cdist)(xyz))

        Optimizable.__init__(self, depends_on=[moving_surface])

    def J(self):
        return self.myJ(
            self.moving_surface.gamma().reshape((-1, 3)),
        )

    def dJ(self):
        return self.myGradJ(
            self.moving_surface.gamma().reshape((-1, 3)),
        )

    return_fn_map = {"J": J, "dJ": dJ}


def deep_copy_surf(
    surface: simsopt.geo.SurfaceRZFourier,
    fprange="field period",
    cartesian_resolution=32,
):
    surface_copy = simsopt.geo.SurfaceRZFourier.from_nphi_ntheta(
        cartesian_resolution,
        cartesian_resolution,
        range=fprange,
        nfp=surface.nfp,
    )
    surface_copy.change_resolution(surface.mpol, surface.ntor)
    surface_copy.rc = surface.rc
    surface_copy.zs = surface.zs
    return surface_copy


def surfgen(
    surface: simsopt.geo.SurfaceRZFourier,
    target_distance: float,
    iterative_constraits=0,
    initial_guess=None,
):
    # Needs to be copied to get the full torus representation,
    # even if the underlying surface was just a half period
    surface_copy = deep_copy_surf(surface)
    if not initial_guess:
        wrapping_surf = deep_copy_surf(surface)
    else:
        wrapping_surf = deep_copy_surf(initial_guess)
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
        jnp.array([target_distance]), jnp.array([1.0]), [surf_surf_dist.J]  # type: ignore
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
            problem.residuals, problem.x.copy(), ftol=1e-5, max_nfev=50  # type: ignore
        )
        problem.x = result.x

        wrapping_surf.change_resolution(wrapping_surf.mpol + 1, wrapping_surf.ntor + 1)
        wrapping_surf.fixed_range(
            0, fourier_resolution, fourier_resolution, fourier_resolution
        )

    return wrapping_surf


def coil_to_surface_distances(
    coil_curves: list[simsopt.geo.CurveRZFourier], surface: simsopt.geo.SurfaceRZFourier
):
    """Computes the signed euclidean distances from discrete points on the coils/curves to the
    respective clostest points on the surface. Positive distance means the curves are outside of the surface.
    """
    if isinstance(coil_curves, list) and not isinstance(
        coil_curves[0], simsopt.geo.CurveRZFourier
    ):
        # If they're not curves, assume theyre coils and convert them to curves.
        curves = [coil.curve for coil in coil_curves]
    else:
        curves = coil_curves

    curve_xyz = jnp.array([c.gamma() for c in curves]).reshape((-1, 3))
    # The deep copy isn't strictly necessary, but we need to ensure that the surface
    # is over the full torus, or the signed distance check may return the wrong sign.
    plasma_coil_distances = -simsopt.geo.signed_distance_from_surface(
        curve_xyz, deep_copy_surf(surface, "full torus", 64)
    )
    return plasma_coil_distances


if __name__ == "__main__":
    import numpy as np

    simple_torus = simsopt.geo.SurfaceRZFourier(5)
    sres = surfgen(simple_torus, 0.1)
    assert abs(sres.minor_radius() - 0.2) < 1e-3
    sres = surfgen(simple_torus, 0.2, iterative_constraits=3)
    assert abs(sres.minor_radius() - 0.3) < 1e-3
    sres = surfgen(simple_torus, 0.3)
    assert abs(sres.minor_radius() - 0.4) < 1e-3
    sres.plot()
