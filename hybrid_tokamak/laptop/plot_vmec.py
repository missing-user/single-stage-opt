import simsopt.mhd
import sys

if __name__ == "__main__": 
    vmec = simsopt.mhd.Vmec(sys.argv[1])
    vmec.boundary.plot()