from . import find_single_l
import os
import simsopt

for top, dirs, files in os.walk("replicate_lgradb/db"):
    for file in files:
        if os.path.splitext(file)[1] == ".json":
            print(file)
            path = os.path.join(top, file)
            surfs, coils = simsopt.load(path)
            res = find_single_l.find_regcoil_distance(surfs[-1])
            print("result=", res)
