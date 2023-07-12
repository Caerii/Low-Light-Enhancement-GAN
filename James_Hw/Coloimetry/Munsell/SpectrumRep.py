import pandas as pd
import numpy as np
import random
import colour
import library
import matplotlib.pyplot as plt
xyz_file = pd.read_csv('xyz_m.csv')
BasisFuncFile = pd.read_excel(library.StdObsFuncsLink, index_col=None)
# print(xyz_file)
lengthXYZdata = len(xyz_file)

sortIt = BasisFuncFile.loc[4:]#.values#.tolist()
sortIt.columns = BasisFuncFile.iloc[3]
newSort = BasisFuncFile.loc[4:].values.tolist()

useful = [i[0:4] for i in newSort]
x_value = [i[0] for i in useful]
arr = []
for i in range(lengthXYZdata):
    num = []
    for j in useful:
        a_val = j[1:4]
        Fstit = xyz_file.loc[i]
        Firstpoint = [Fstit['x'],Fstit['y'],Fstit['z']]
        res = np.sum(np.multiply(Firstpoint,a_val))#np.convolve()
        num.append(res)
        # x = Fstit['x']
        # y = Fstit['y']
        # z = Fstit['z']
    arr.append(num)
    # arr.append([i,num])
for i in arr:
    plt.plot(x_value,i)
plt.show()
# print(sortIt)
# print(newSort)
# print(useful)
# print(arr)
