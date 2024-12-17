import simsopt
from simsopt import mhd
import simsopt.geo
import numpy as np
import sys
sys.path.append("/home/missinguser/CSE/single-stage-opt/")
from quasr_coil_check import bdistrib_io
import subprocess

def generate_Bn_initial(rotating_ellipse = True, plot=False):
    if rotating_ellipse:
        equil = mhd.Spec("hybrid_tokamak/laptop/rotating_ellipse_fb_low.sp", verbose=True)
    else:
        equil = mhd.Spec("hybrid_tokamak/laptop/nfp2_QA_iota0.4.sp", verbose=True)
    hybrid_surface = equil.boundary.copy()
    middle_surface = equil.boundary.copy(range="field period")
    # Called outers instead of outer to align with middle & hybrid <3
    outers_surface = equil.boundary.copy(range="field period")

    if rotating_ellipse:
        # Make a circular torus
        middle_surface = equil.computational_boundary
        middle_surface.change_resolution(equil.mpol, equil.ntor)
        
        outers_surface =  equil.computational_boundary.copy(range="field period")
        outers_surface.change_resolution(equil.mpol, equil.ntor)
        outers_surface.scale(1.2)
        outers_surface.set_rc(0, 0, middle_surface.get_rc(0, 0) )
    else:
        scaleR0 = 0.98
        # Make an ellipsoid shape
        middle_surface.change_resolution(hybrid_surface.mpol, 0)
        middle_surface.change_resolution(equil.mpol, equil.ntor)
        middle_surface.set_rc(0,1, hybrid_surface.get_rc(0,1)) 
        
        middle_surface.scale(2.4)
        middle_surface.set_rc(0, 0, hybrid_surface.get_rc(0, 0)*scaleR0 )

        outers_surface.change_resolution(hybrid_surface.mpol, 0)
        outers_surface.change_resolution(equil.mpol, equil.ntor)
        outers_surface.set_rc(0,1, hybrid_surface.get_rc(0,1)) 
        outers_surface.scale(2.5)
        outers_surface.set_rc(0, 0, hybrid_surface.get_rc(0, 0)*scaleR0)

    def plot_scalar_surf(surf:simsopt.geo.SurfaceRZFourier, scalar, ax=None, **kwargs):
        gamma = surf.gamma()
        
        # plot in matplotlib.pyplot
        import matplotlib.pyplot as plt
        from matplotlib import cm
        my_col = cm.jet(scalar)
        surf.get_quadpoints(scalar.shape[0], scalar.shape[1], nfp=surf.nfp)

        if ax is None or ax.name != "3d":
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2], facecolors = my_col, **kwargs)
        plt.show()

    hybrid_surface = simsopt.geo.SurfaceRZFourier.from_vmec_input("hybrid_tokamak/laptop/fixedb/QIlowres/input.rot_ellipse_000_000159")

    if plot and equil.freebound:
        simsopt.geo.plot(
            [
                hybrid_surface,
                equil.computational_boundary
            ],
            engine="plotly",
        )

    # if plot:
    #     simsopt.geo.plot(
    #         [
    #             hybrid_surface,
    #             middle_surface,
    #             outers_surface,
    #         ],
    #         engine="plotly",
    #     )

    bdistrib_io.write_netcdf("hybrid_tokamak/wout_nfp2_QA_iota0.4.nc", hybrid_surface, "hybrid_tokamak/wout_nfp2_QA_iota0.4_000_000001.nc")
    bdistrib_io.write_nescin_file("hybrid_tokamak/nescin.msurf", middle_surface)
    bdistrib_io.write_nescin_file("hybrid_tokamak/nescin.osurf", outers_surface)

    subprocess.check_call(
        [
            "../../regcoil/regcoil",
            "regcoil_in.hybrid_tokamak",
        ],
        cwd="hybrid_tokamak",
    )


    from scipy.io import netcdf_file
    import matplotlib.pyplot as plt

    filename = "hybrid_tokamak/regcoil_out.hybrid_tokamak.nc"
    f = netcdf_file(filename, "r", mmap=False)
    nfp = f.variables["nfp"][()]
    ntheta_plasma = f.variables["ntheta_plasma"][()]
    nzeta_plasma = f.variables["nzeta_plasma"][()]
    theta_plasma = f.variables["theta_plasma"][()]
    zeta_plasma = f.variables["zeta_plasma"][()]
    Bnormal_total_middle = f.variables["Bnormal_total_middle"][()]
    
    BdotN_fft = np.fft.fft2(Bnormal_total_middle)
    if plot:
        plt.figure(figsize=(16,5))
        plt.subplot(131)
        plt.imshow(Bnormal_total_middle)
        plt.title("BdotN")
        plt.colorbar()
        plt.subplot(132)
        plt.imshow(np.real(np.fft.fftshift(BdotN_fft)))
        plt.title("fft real")
        plt.colorbar()
        plt.subplot(133)
        plt.imshow(np.imag(np.fft.fftshift(BdotN_fft)))
        plt.title("fft imag")
        plt.colorbar()
        plt.show()
        print(BdotN_fft.shape)
    exit()
    import py_spec
    nml = py_spec.SPECNamelist(equil.extension+".sp")
    
    # Force updating the internal arrays since mpol!=equil.mpol
    nml.update_resolution(equil.mpol-1, equil.ntor)
    nml.update_resolution(equil.mpol, equil.ntor)
    hybrid_surface.change_resolution(equil.mpol, equil.ntor)
    middle_surface.change_resolution(equil.mpol, equil.ntor)

    N = BdotN_fft.shape[0]*BdotN_fft.shape[1]
    BdotN_fft /= N
    scaling = 7.012800640000000E-02/-0.04110736474316339
    BdotN_fft *= scaling
    for m in range(equil.mpol):
        nmin = -equil.ntor
        if m == 0:
            nmin = 0
        for n in range(nmin, equil.ntor):
            # equil.normal_field.set_vns(m, n, BdotN_fft[n, m])
            nml['physicslist']["Rbc"][m][n+nml._Ntor] = hybrid_surface.get_rc(m, n)
            nml['physicslist']["Zbs"][m][n+nml._Ntor] = hybrid_surface.get_zs(m, n) 
            nml['physicslist']["Rwc"][m][n+nml._Ntor] = middle_surface.get_rc(m, n)
            nml['physicslist']["Zws"][m][n+nml._Ntor] = middle_surface.get_zs(m, n) 
            nml['physicslist']["Vns"][m][n+nml._Ntor] = np.imag(BdotN_fft[n, m])
    # nml.write(equil.extension+"_Vns.sp")
    nml.write(equil.extension+".sp", force=True)

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(equil.lib.inputlist)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    # BdotN_fft.shape[0] = nzeta_plasma (n tor)
    # BdotN_fft.shape[1] = ntheta_plasma (m pol)
    return hybrid_surface, middle_surface, np.imag(BdotN_fft)

if __name__ == "__main__":
    generate_Bn_initial(rotating_ellipse=True, plot=True)