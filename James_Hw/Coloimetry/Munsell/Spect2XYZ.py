import pandas as pd
import numpy as np
import random
import colour
import library
import matplotlib.pyplot as plt
spectrum_file = pd.read_csv('spectrum_m.csv')
BasisFuncFile = pd.read_excel(library.StdObsFuncsLink, index_col=None)
lengthSpecdata = len(spectrum_file)
SortIt_pre = BasisFuncFile.iloc[:, 0:4]
sortIt = SortIt_pre.loc[4:]#.values#.tolist()
sortIt.columns = SortIt_pre.iloc[3]
# sortIt.index = 
# newSort = BasisFuncFile.loc[4:].values.tolist()
indexTitle = ['xbar','ybar','zbar']
# Wavelength (nm)      xbar      ybar  ...      xbar      ybar      zbar
x_value_pre = spectrum_file.columns
x_value = x_value_pre[1:]
# print(sortIt)

f = open('result.txt','w')

for i in range(lengthSpecdata):
    clr_ra = []
    for j in range(3):
        sumnum = 0
        for k in x_value:
            # print(type(k))
            intent = sortIt.loc[sortIt['Wavelength (nm)'] == int(k)]
            num = intent[indexTitle[j]].values
            df = spectrum_file.loc[i,k]
            rt = df * num[0]
            sumnum += rt
            # sortIt.where()
            # intent[]
        clr_ra.append(sumnum)
    ft = ''
    for i in clr_ra:
        ft += str(i) + ' '
    ft += '\n'
    f.write(ft)
f.close()
