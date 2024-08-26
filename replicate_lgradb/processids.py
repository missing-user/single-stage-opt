import numpy as np
import precompute_regcoil
ids= np.loadtxt("representative_subset/IDs.txt")
for i in ids:
  precompute_regcoil.compute_and_save(int(i))