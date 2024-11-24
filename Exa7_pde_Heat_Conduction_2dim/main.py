import os
import subprocess

import shutil
Loop=10

pwd=os.path.abspath(__file__)
par_pwd=os.path.dirname(pwd)
file_path=par_pwd+'/experiment.py'

data_path = par_pwd+'/experimentData'
if os.path.exists(data_path):
    shutil.rmtree(data_path)

# os.system(file_path)
command='python '+file_path

for i in range(Loop):
    subprocess.call(command, shell=True)

file_path=par_pwd+'/Draw.py'
command='python '+file_path
subprocess.call(command, shell=True)
