
import subprocess
import os
class SpecRename(object):
    """
    Helper class to deal with the .sp.end and .sp.h5 files. 
    This class will temporarily copy the .sp.end and .sp.h5 files to .sp and delete them on exit from the function. 

    Usage:
    with SpecRename(filename) as specf:
        spec = mhd.Spec(specf)
    """
    def __init__(self, filename):
        self.have_to_delete = False
        if filename.endswith(".sp"):
            self.filename = filename
            return
        elif filename.endswith(".sp.end"):
            sp_file = filename[:-4]
            if not os.path.exists(sp_file):
                subprocess.check_call(["cp", filename, sp_file])
                self.have_to_delete = True
            self.filename = sp_file
            return
        elif filename.endswith(".sp.h5"):
            sp_file = filename[:-3]
            end_file =  sp_file + ".end"
            if os.path.exists(end_file):
                if not os.path.exists(sp_file):
                    subprocess.check_call(["cp", end_file, sp_file])
                    self.have_to_delete = True
                self.filename = sp_file
                return
            else:
                raise ValueError(f"File {end_file} does not exist")
        raise ValueError("filename should end with .sp, .sp.end or .sp.h5")
    def __enter__(self):
        return self.filename
    def __exit__(self, type, value, traceback):
        if self.have_to_delete:
            subprocess.check_call(["rm", self.filename])
