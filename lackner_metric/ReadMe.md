Investigating whether the integral over the virtual-casing surface currents is a good indicator of the currents required in the coils. 
This effort was motivated by a discussion with Karl Lackner, but not continued beyond some initial exploration due to lack of time.

Requires a modified version of REGCOIL, which supports a `middle` surface, on which a normal magnetic field is computed, based on the current density distribution on the outer surface. https://github.com/missing-user/regcoil/tree/regcoil-for-ipp

The scripts are the only ones in this repository to use DESC instead of simsopt, because I wanted to try it out.
