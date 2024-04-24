# This script should be run outside of the virutal environment, using the command pvpython

from paraview import simple 
import numpy as np
import time

reader_s = simple.OpenDataFile("output/surf_opt.vts")
rep = simple.Show(reader_s)

# Set background color to black
camera = simple.GetActiveCamera()
display = simple.GetDisplayProperties()
view = simple.GetActiveView()
simple.SetViewProperties(Background=[0.0, 0.0, 0.0],
                          UseColorPaletteForBackground = 0)
# Color by the parameter B_N
simple.ColorBy(rep, "B_N")
rep.RescaleTransferFunctionToDataRange(True)
display.SetScalarBarVisibility(view, True)

# Read the curve files for the coils
reader_c = simple.OpenDataFile("output/curves_opt.vtu")
rep_c = simple.Show(reader_c)


#simple.Interact()

init_pos = camera.GetPosition()
angle = 0.0
while True:
  start_time = time.time()
  camera.SetPosition((np.sin(angle)*np.linalg.norm(init_pos), 0, np.cos(angle)*np.linalg.norm(init_pos)))
  simple.Show()
  simple.Render()
  #simple.ReloadFiles(reader_s)
  simple.ReloadFiles(reader_c)
  rep_c.RescaleTransferFunctionToDataRange(True)

  dt = time.time() - start_time
  #angle += np.deg2rad(20*dt)