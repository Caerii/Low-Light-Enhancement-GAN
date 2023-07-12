import numpy as np
from numpy import fft
import scipy
from scipy import misc, fftpack
import cv2
from matplotlib import pyplot as plt
import pywt
def extract_dwt_single_level(mod):
    # coeff_sort = np.sort(np.ndarray.flatten(np.abs(mod)))[::-1]
    # length = len(coeff_sort)
    # ext_len = length//10+int(bool(length%10))
    # res_coeff= coeff_sort[:ext_len]
    # print(res_coeff)
    # thres = min(list(res_coeff))
    blank = np.ndarray.flatten(np.abs(mod))
    thres = thres_val_comb(blank)
    mod[np.abs(mod) < thres] = 0
    return mod
    # return res_coeff
def extract_dwt_multi_level(mid):
    br = []
    arr = []
    coeff_Ext = mid[1]#np.array(mid[1:])
    for i in coeff_Ext:
        ite = extract_dwt_single_level(i)
        arr.append(ite)
    first = np.array(mid[0])
    rde = extract_dwt_single_level(first)
    br.append(rde)
    br.append(arr)
    return br
    # return arr
# def extract_SingleLayer():
def PrintSq(it):
    for i in it:
        print(i)
def thres_val_comb(blank):
    sort_blank = np.sort(blank)[::-1]
    length = len(sort_blank)
    ext_len = length//10+int(bool(length%10))
    res_coeff= sort_blank[:ext_len]
    thres = np.min(list(res_coeff))
    return thres
def flatten_mulLvl(mid):
    ite = []
    coeff_first = mid[0]
    coeff_Ext = mid[1:]#np.array()
    blank = np.ndarray.flatten(np.abs(coeff_first))
    ite.extend(blank)
    # res_arr = 0
    for level_item in coeff_Ext:
        for i in level_item:
            blank = np.ndarray.flatten(np.abs(i))
            ite.extend(blank)     
    # print(coeff_Ext)
    # print(len(mid))
    # print(len(coeff_Ext))
    return ite
def handler_first(it,t):
    ite = []
    for i in it:
        itr = []
        for j in i:
            if np.abs(j)>=t:
                itr.append(j)
            else:
                itr.append(0)
        ite.append(itr)
    return ite
def handler_Back(it,t):
    ite = []
    for i in it:
        itr = []
        for j in i:
            ita = []
            for k in j:
                if np.abs(k)>=t:
                    ita.append(k)
                else:
                    ita.append(0)
            itr.append(ita)
        ite.append(itr)
    return ite

def cof_ext_wavedec2(mid):
    ite = []
    coeff_first = mid[0]
    coeff_Ext = mid[1:]
    itr = flatten_mulLvl(mid)
    thres = thres_val_comb(itr)
    # ite = pywt.threshold(mid,thres,'hard',substitute=0)
    print(thres)
    coeff_first = handler_first(coeff_first,thres)
    # coeff_first[np.abs(coeff_first) < thres] = 0
    # PrintSq(coeff_first)
    ite.append(coeff_first)
    for level_item in coeff_Ext:
        rak = []
        for dir_it in level_item:
            # print(dir_it.shape)
            tu = handler_Back(dir_it,thres)
            rak.append(tu)
        newarr = tuple(rak)
        # level_item[np.abs(level_item) < thres] = 0
        ite.append(newarr)
    # mid[np.abs(mid)< thres] = 0
    return ite
    # return mid
# def coeff_extractor(mid):
    
#     # coeff_non_sort = np.ndarray.flatten(coeff_Ext)
#     # print(coeff_non_sort)
#     coeff_sort = np.sort(coeff_non_sort)[::-1]
#     length = len(coeff_sort)
#     ext_len = length//10+int(bool(length%10))#round(length/10)
#     res_coeff= coeff_sort[:ext_len]
#     return res_coeff

input_imageLenna = cv2.imread('lena512x512.png')
print(input_imageLenna.shape)
# mid_lena = pywt.dwt2(input_imageLenna,'db1')
# mid_lena_2 = pywt.swt2(input_imageLenna,'db1')
mid_lena = pywt.wavedec2(input_imageLenna,'db1',level=3)

cof_lena = cof_ext_wavedec2(mid_lena)

# print()
# # print(output)
# cof_lena = extract_dwt_multi_level(mid_lena)
# cof_lena_2 = extract_dwt_multi_level(mid_lena_2)
# cof_lena = coeff_extractor(mid_lena)
# output = pywt.idwt2(cof_lena,'db1')
# output = pywt.idwt2(mid_lena,'db1')
# output2 = pywt.idwt2(mid_lena_,'db1')

output = pywt.waverec2(cof_lena,'db1')
# plt.imshow(input_imageLenna)
print(output.shape)
plt.imshow(output)
plt.show()