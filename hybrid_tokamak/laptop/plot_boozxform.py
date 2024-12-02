from simsopt import mhd
import matplotlib.pyplot as plt
import numpy as np

import booz_xform
import matplotlib.pyplot as plt
import numpy as np

import sys

def boozer_plot(filename):
    b1 = booz_xform.Booz_xform()
    b1.read_wout(filename)
    b1.compute_surfs = [5, 64]
    b1.run()

    plt.subplot(1, 2, 1)
    booz_xform.surfplot(b1, js=0)
    plt.subplot(1, 2, 2)
    booz_xform.surfplot(b1, js=0, fill=False)
    plt.figure()

    plt.subplot(1, 2, 1)
    booz_xform.surfplot(b1, js=1)
    plt.subplot(1, 2, 2)
    booz_xform.surfplot(b1, js=1, fill=False)
    # plt.figure()
    # booz_xform.symplot(b1)
    plt.show()

if __name__ == "__main__":
    boozer_plot(sys.argv[1])  