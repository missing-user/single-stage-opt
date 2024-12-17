from simsopt import field, geo, mhd, objectives

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

import sys
from spec_rename import SpecRename

if __name__ == "__main__":
    for filename in sys.argv[1:]: 
        with SpecRename(filename) as specf:
            spec = mhd.Spec(specf)
        coilsurf = spec.boundary.copy()
        coilsurf = spec.computational_boundary.copy()
        coilset = field.CoilSet.for_surface(coilsurf, coils_per_period=4, current_constraint='fix_all')
        # coilset.plot()
        # plt.imshow(coilsurf.inverse_fourier_transform_scalar(spec.normal_field.vns, np.zeros_like(spec.normal_field.vns), normalization=(2*np.pi)**2, stellsym=spec.stellsym))
        # plt.colorbar()
        # plt.show(block=False)
        normal_field = field.normal_field.CoilNormalField(coilset)

        # result = normal_field.optimize_coils(spec.normal_field.vns, TARGET_LENGTH=100 * 2*np.pi* spec.computational_boundary.minor_radius(), MAXITER=950)  
        coilset = field.CoilSet.for_surface(spec.boundary, coils_per_period=4, current_constraint='fix_all')
        result = normal_field.optimize_coils(np.zeros_like(spec.normal_field.vns), TARGET_LENGTH=32 * 2*np.pi* spec.computational_boundary.minor_radius(), MAXITER=450)  
        print(result)

        geo.plot([spec.boundary, spec.computational_boundary.copy(range="field period")]+normal_field.coilset.coils, engine="plotly")
        
        nfieldlines = 32
        R1 = spec.computational_boundary.minor_radius()
        R0 = spec.boundary.major_radius()
        print(f"R0={R0}, R1={spec.boundary.minor_radius()}")
        rr = np.linspace(R0, R0+R1/2, nfieldlines)
        Z0 = np.zeros(nfieldlines)

        n = 60
        degree = 4
        rrange = (R0, R1, n) 
        phirange = (0, 2*np.pi/spec.nfp, 4*n*2)
        # exploit stellarator symmetry and only consider positive z values:
        zrange = (0, R1, n//2)
        bsh = field.InterpolatedField(
            normal_field.coilset.bs, degree, rrange, phirange, zrange, True, nfp=spec.nfp, stellsym=True
        )
        print("Constructed fieldline interpolant")
        phis = [(i/4)*(2*np.pi/spec.nfp) for i in range(4)]
        fieldlines_tys, fieldlines_phi_hits = field.compute_fieldlines(
            bsh, rr, Z0, tol=1e-14, tmax=10000,
            phis=phis, stopping_criteria=[
                field.MinRStoppingCriterion(R0-R1),
                field.MaxRStoppingCriterion(R0+R1), 
                field.IterationStoppingCriterion(100000),
                                          ])
        print(f"Time for fieldline. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
        field.plot_poincare_data(fieldlines_phi_hits, phis, f'poincare_fieldline_.png', dpi=650) 

