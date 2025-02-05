#!/bin/bash

source .venv/bin/activate
python -m lackner_metric.generate_two_surfaces

scp {nescin.msurf,nescin.osurf,lackner_metric/regcoil_in.hybrid_tokamak} juph@cluster:/home/IPP-HGW/juph/regcoil/
scp lackner_metric/wout_NAS.nv.n4.SS.iota43.Fake-ITER.01_000_000000.nc juph@cluster:/home/IPP-HGW/juph/regcoil/

ssh cluster 'ml intel mkl ompi hdf5 netcdf fftw && cd regcoil && ./regcoil regcoil_in.hybrid_tokamak'

scp juph@cluster:/home/IPP-HGW/juph/regcoil/regcoil_out.hybrid_tokamak.nc ./lackner_metric/
python ../regcoil/regcoilPlot.py lackner_metric/regcoil_out.hybrid_tokamak.nc
