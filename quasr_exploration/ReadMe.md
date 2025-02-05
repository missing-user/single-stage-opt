# Overview

## Pre-processing Utilities
Due to the large amount of data in QUASR, my data processing pipeline consists of various pre-processing scripts for the individual tasks (e.g. computing regcoil for each equilibrium, or generating a winding surface) that can mostly be run in parallel on different machines.
- `precompute_complexities.py`: Computes various coil complexity metrics of the filament coils.
- `precompute_surfaces.py`: Computes the offset winding surfaces for each equilibrium using the `surfgen.py` script.
- `precompute_bdistrib.py`: Computes the efficiency sequences for each equilibrium using the `bdistrib` code, based on the precomputed winding surfaces.
- `precompute_regcoil.py`: Computes the current density distributions for each equilibrium (NOT L_{REGCOIL}, that is in the lgradb_quasr_analysis) for a fixed winding surface offset.

## Analysis Utilities
- `quasr_explorer.py`: Interactive tool for exploring coil complexity, efficiency sequences and geometries of configurations in the QUASR database. Different quantities can be plotted against each other to facilitate finding correlations and interesting relationships.
- `quasr.ipynb`: Intial poking around on the data, and some analysis of individual results.

## Individual Plotting Utilities 
- `bdistribPlot.py`: Plots the efficiency sequences and other quantities from the `bdistrib` output. Only slightly modified from the original plotting script checked into `bdistrib` by Matt Landreman.
- `simsoptPlot.py`: Plots the surfaces and coils from a `simsopt` json file, like the ones in QUASR.
- `regcoilPlot.py`: Plots the cross sections and current density distributions from a `regcoil` output file.

