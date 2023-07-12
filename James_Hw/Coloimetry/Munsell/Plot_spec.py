import pandas as pd
import numpy as np
import random
import colour
import library
import matplotlib.pyplot as plt
spectrum_file = pd.read_csv('spectrum_m.csv')
# print(spectrum_file[0])
x_value_pre = spectrum_file.columns
x_value = x_value_pre[1:]
arr = [i[1:] for i in spectrum_file.values.tolist()]
for i in arr:
    plt.plot(np.array(x_value),np.array(i))
plt.show()