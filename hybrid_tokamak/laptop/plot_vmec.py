import simsopt.mhd
import sys
import numpy as np
import matplotlib.pyplot as plt
if __name__ == "__main__": 
    vmecs = [simsopt.mhd.Vmec(filename) for filename in sys.argv[1:]]
    # for phi in [0, np.pi/2, np.pi]:
    for vmec in vmecs:
        cross = vmec.boundary.cross_section(0)
        plt.plot(cross[:,0], cross[:,2])
    plt.legend(sys.argv[1:])
    plt.show()
    for filename in sys.argv[1:]:
        vmec = simsopt.mhd.Vmec(filename)
        # vmec.boundary.plot(show=False)
        vmec.boundary.scale(4)
        vmec.boundary.plot(show=False)
    plt.show()

    