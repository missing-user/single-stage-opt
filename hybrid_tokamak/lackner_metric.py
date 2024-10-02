import numpy as np
import io
import f90nml

MU0 = 4e-7 * np.pi


def generate_non_perturbed_nml(input_nml):
    """Parse VMEC input namelist, set all toroidal modes to zero, write the output to a new namelist called <input name>.unperturbed"""
    with open(input_nml, "r") as f:
        txt = f.readlines()
        # Remove the last line containing "&END" because f90nml can't deal with it
        if txt[-1] == "&END":
            nml_file = io.StringIO()
            for line in txt[:-1]:
                nml_file.write(line)
            nml_file.seek(0)
        else:
            nml_file = f
        nml = f90nml.read(nml_file)
        nml.uppercase = True
        print(nml["indata"]["zbs"])
        # nml["indata"]["ntor"]
    nml.write(input_nml + ".unperturbed", force=True)


def compute():
    # Cartesian magnetic field
    B_xyz = 1
    # Unit normal vector
    n = 1
    K = np.cross(B_xyz, n, axis=-1) / MU0


if __name__ == "__main__":
    import sys

    generate_non_perturbed_nml(sys.argv[1])
