Repository for my master thesis **novel approaches to combined plasma-coil optimization** at [IPP](https://www.ipp.mpg.de/) with [Sophia Henneberg](https://sahenneberg.wordpress.com/) 

## Abstract
Stellarator design is traditionally performed using a two-stage approach, with the first stage optimizing a plasma equilibrium for the desired properties, like good plasma confinement, and the second stage focuses on finding a set of coils to produce the required magnetic field while satisfying engineering constraints. 
Considering coil complexity early during the first-stage optimization is crucial to improve the feasibility and performance of the final design. 
This thesis explores different methods for quantifying and improving coil complexity, especially plasma-coil-distance, during first-stage optimization.
In particular, we investigate whether coil complexity can be reduced by choosing different degrees of freedom for the first-stage optimization. The external, normal magnetic field on a computational boundary is used as the degree of freedom for stellarator optimization (quasi-free boundary optimization), which is performed using the Stepped Pressure Equilibrium Code (SPEC). 
Comparison of the resulting quasi-free-boundary optimized equilibria with those from classical fixed-boundary optimizations does not reveal any benefits in terms of the attainable plasma-coil distance and quasi-symmetry fulfillment. 
We contribute a new implementation of the virtual-casing integration routine to SPEC, which boasts improved numerical accuracy and speeds up the computation of free-boundary stellarator equilibria by approximately a factor of \SI{20}{}. 
This thesis further includes the systematic validation of both a recently developed proxy for plasma-coil distance, and a novel proxy based on the theory of efficient fields, on a large dataset of quasi-symmetric vacuum equilibria. 

## Overview
