import colour 
import cv2
import numpy as np 
img = cv2.imread('C:/Users/Josep/OneDrive/Desktop/DTU course/day 4/34269_AWB/Blue cast.jpg')


#calculate the average color
avg_color_row = np.average(img,axis = 0)
avg_color = np.average(avg_color_row, axis =0)
avg_red = np.average(avg_color[0])
avg_green = np.average(avg_color[1])
avg_blue = np.average(avg_color[2])
redf = avg_green/ avg_red
bluef = avg_green/avg_blue
rows,cols,colourNum = img.shape
img_white_balanced = img.copy()
for i in range(rows):
    for j in range(cols):
        img_white_balanced[i][j][0] = img[i][j][0] * redf
        img_white_balanced[i][j][2] = img[i][j][2] * bluef

# new_red = redf * avg_red
# new_blue = bluef * avg_blue

# img_white_balanced = cv2.convertScaleAbs(img, alpha=new_red, beta=avg_green)

# Display the original and white balanced images side by side
#cv2.imshow('Original', img)
#cv2.imshow('White Balanced', img_white_balanced)
from PIL import Image



max_red = np.amax(img[:,:,0])
max_green = np.amax(img[:,:,1])
max_blue = np.amax(img[:,:,2])
redf2 = max_green/ max_red
bluef2 = max_green / max_blue
img_white_balanced2 = img.copy()
for i in range(rows):
    for j in range(cols):
        img_white_balanced2[i][j][0] = img[i][j][0] * redf2
        img_white_balanced2[i][j][2] = img[i][j][2] * bluef2
# Display the original and white balanced images side by side
original_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
white_balanced_image = Image.fromarray(cv2.cvtColor(img_white_balanced, cv2.COLOR_BGR2RGB))
white_balanced_image2 = Image.fromarray(cv2.cvtColor(img_white_balanced2, cv2.COLOR_BGR2RGB))
original_image.show(title='Original')
white_balanced_image.show(title='White Balanced')
white_balanced_image2.show(title='White Balanced2')