Repository for my master thesis **novel approaches to combined plasma-coil optimization** at [IPP](https://www.ipp.mpg.de/) with [Sophia Henneberg](https://sahenneberg.wordpress.com/) 

## Abstract
Stellarator design is traditionally performed using a two-stage approach, with the first stage optimizing a plasma equilibrium for the desired properties, like good plasma confinement, and the second stage focuses on finding a set of coils to produce the required magnetic field while satisfying engineering constraints. 
Considering coil complexity early during the first-stage optimization is crucial to improve the feasibility and performance of the final design. 
This thesis explores different methods for quantifying and improving coil complexity, especially plasma-coil-distance, during first-stage optimization.
In particular, we investigate whether coil complexity can be reduced by choosing different degrees of freedom for the first-stage optimization. The external, normal magnetic field on a computational boundary is used as the degree of freedom for stellarator optimization ([quasi-free boundary optimization](https://doi.org/10.1017/S0022377821000271 )), which is performed using the Stepped Pressure Equilibrium Code ([SPEC](https://github.com/PrincetonUniversity/SPEC)). 
Comparison of the resulting quasi-free-boundary optimized equilibria with those from classical fixed-boundary optimizations does not reveal any benefits in terms of the attainable plasma-coil distance and quasi-symmetry fulfillment. 
We contribute a new implementation of the virtual-casing integration routine to SPEC, which boasts improved numerical accuracy and speeds up the computation of free-boundary stellarator equilibria by approximately a factor of 20. 
This thesis further includes the systematic validation of both a recently developed proxy for plasma-coil distance, and a novel proxy based on the theory of efficient fields, on a large dataset of quasi-symmetric vacuum equilibria. 

## Overview
- `plots/` most simple scripts to generate the plots for my thesis can be found here. More complex plotting were sometimes written as part of the main scripts of the respective tasks instead.
- `quasr_exploration` Interactive tool for exploring coil complexity, efficiency sequences and geometries of configurations in the QUASR database. 
- `lgradb_quasr_analysis` validation of $L_{\nabla B}^*$ and related the magnetic gradient length scale on a bigger dataset of stellarators. Evaluation of the theory of efficient fields. 
- `synthetic_normal_fields` Exploring whether the Fourier spectrum of the normal magnetic field on a simple boundary can be correlated with coil complexity 
- `qfb_optimization` refers to quasi-free-boundary, so stellarator optimization using the normal magnetic field as the degree of freedom. 
Folders starting with `fixb` or `freeb` contain the quasi free boundary optimization results at various points throughout development. 
Similar results can be optained by running `qfb_optimization/spec_fb_opt.py`.
