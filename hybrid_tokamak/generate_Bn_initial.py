import simsopt
from simsopt import mhd
import simsopt.geo
import numpy as np
from quasr_coil_check import bdistrib_io
import subprocess

rotating_ellipse = False
if rotating_ellipse:
    equil = mhd.Spec("hybrid_tokamak/laptop/rotating_ellipse_fb.sp", verbose=True)
else:
    equil = mhd.Spec("hybrid_tokamak/laptop/nfp2_QA_iota0.4.sp", verbose=True)

equil.boundary.plot(engine="plotly")

hybrid_surface = equil.boundary.copy()
middle_surface = equil.boundary.copy()
# Called outers instead of outer to align with middle & hybrid <3
outers_surface = equil.boundary.copy()

if rotating_ellipse:
    # Make a circular torus
    middle_surface.change_resolution(middle_surface.mpol, 0)
    middle_surface.scale(1.6)
    middle_surface.set_rc(0, 0, hybrid_surface.get_rc(0, 0) )

    outers_surface.change_resolution(outers_surface.mpol, 0)
    outers_surface.scale(1.8)
    outers_surface.set_rc(0, 0, hybrid_surface.get_rc(0, 0) )
else:
    # Make an ellipsoid shape
    middle_surface.change_resolution(middle_surface.mpol, 0)
    middle_surface.change_resolution(outers_surface.mpol, 1)
    middle_surface.set_rc(0,1, hybrid_surface.get_rc(0,1))

    middle_surface.scale(2)
    middle_surface.set_rc(0, 0, hybrid_surface.get_rc(0, 0) )

    outers_surface.change_resolution(outers_surface.mpol, 0)
    outers_surface.change_resolution(outers_surface.mpol, 1)
    outers_surface.set_rc(0,1, hybrid_surface.get_rc(0,1)) 
    outers_surface.scale(2.1)
    outers_surface.set_rc(0, 0, hybrid_surface.get_rc(0, 0) )

simsopt.geo.plot(
    [
        hybrid_surface,
        middle_surface,
        outers_surface,
    ],
    engine="plotly",
)

bdistrib_io.write_netcdf("hybrid_tokamak/wout_nfp2_QA_iota0.4.nc", hybrid_surface)
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
numContours = 10
ntheta_plasma = f.variables["ntheta_plasma"][()]
nzeta_plasma = f.variables["nzeta_plasma"][()]
theta_plasma = f.variables["theta_plasma"][()]
zeta_plasma = f.variables["zeta_plasma"][()]
Bnormal_total_middle = f.variables["Bnormal_total_middle"][()]
plt.contourf(
    zeta_plasma,
    theta_plasma,
    np.transpose(Bnormal_total_middle),
    numContours,
)


BdotN_fft = np.fft.fft2(Bnormal_total_middle)

# px.imshow(BdotN).show()
# px.imshow(np.abs(np.fft.fftshift(BdotN_fft))).show()
# px.imshow(np.real(np.fft.fftshift(BdotN_fft)), title="Real component").show()
# px.imshow(np.imag(np.fft.fftshift(BdotN_fft)), title="Imag component").show()
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
# plt.tight_layout()
plt.show()

print()