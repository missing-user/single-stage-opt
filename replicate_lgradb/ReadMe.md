# Investigating the normalization for LgradB

In the paper the authors validate the performance of the LgradB metric by comparing it to the REGCOIL complexity of the configuration. This analysis uses the following normalization for REGCOIL:

    For the results that follow, we chose $\lambda$ so that $B_{RMS} = 0.01 T$, with all configurations scaled to an average field strength in the plasma of $B_0 = 5.865 T$.

where $B_0$ is the volume average magnetic field. It is unclear why the volume average would be chosen here instead of e.g. maximum field on axis.

## Python script to compare analytic derivation for B_volume average with numerical results of shaped stellerators
`precise_qi_Bvolume.ipynb`
