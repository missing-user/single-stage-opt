import plotly
import simsopt
import simsopt.geo
import sys

if len(sys.argv) != 2:
    print("Usage: ", sys.argv[0], " <path to simsopt file>")
path = sys.argv[1]
surfs, coils = simsopt.load(path)
# surfs[-1].plot(close=True, show=True)
simsopt.geo.plot(surfs + coils, close=True, engine="plotly")
