import glob
import shutil
import time


"""
Duplicated Code
"""


for filename in glob.glob(r"./S4/1113M2R/*"):
    shutil.copy(filename,r"./S4/annotation/Nerve/Mix_"+filename.split("/")[-2]+"__"+filename.split("/")[-1])