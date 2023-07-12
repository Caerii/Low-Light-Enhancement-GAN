import pandas as pd
import numpy as np
import random
import colour
import library
# spectrum_data = pd.read_csv('spectrum_m.csv')
xyzdata = pd.read_csv('xyz_m.csv')
lengthXYZdata = len(xyzdata) 
chosen = random.choices(range(lengthXYZdata),k=2)
Fstit = xyzdata.loc[chosen[0]]
Sndit = xyzdata.loc[chosen[1]]
Firstpoint = [Fstit['x'],Fstit['y'],Fstit['z']]
Secondpoint = [Sndit['x'],Sndit['y'],Sndit['z']]
# print(Firstpoint,Secondpoint)
FstLab = colour.XYZ_to_Lab(Firstpoint)
SecLab = colour.XYZ_to_Lab(Secondpoint)
num = colour.delta_E(FstLab, SecLab,method='CIE 1976')
print(num)
