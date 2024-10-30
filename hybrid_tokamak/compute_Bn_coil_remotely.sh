#!/bin/bash

source .venv/bin/activate
python -m hybrid_tokamak.generate_two_surfaces

scp hybrid_tokamak/{nescin.msurf,nescin.osurf,regcoil_in.hybrid_tokamak} juph@cluster:/home/IPP-HGW/juph/regcoil/
scp hybrid_tokamak/wout_NAS.nv.n4.SS.iota43.Fake-ITER.01_000_000000.nc juph@cluster:/home/IPP-HGW/juph/regcoil/

ssh cluster 'ml intel mkl ompi hdf5 netcdf fftw && cd regcoil && ./regcoil regcoil_in.hybrid_tokamak'

scp juph@cluster:/home/IPP-HGW/juph/regcoil/regcoil_out.hybrid_tokamak.nc ./hybrid_tokamak/
python ../regcoil/regcoilPlot.py hybrid_tokamak/regcoil_out.hybrid_tokamak.nc
