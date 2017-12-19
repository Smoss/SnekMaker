import cv2
import os
import shutil

snakes = os.listdir('.')
x = 0
for file in snakes:
	if file != 'snekResizer.py' and file != "Squares2" and file != "Squares":
		image = cv2.imread(file)
		r = 100.0 / image.shape[1]
		dim =(32, 32)
		newimg = cv2.resize(image, dim)
		if x%10 == 0:
			cv2.imwrite("Squares2\\"+file + ".png", newimg)
		else:
			cv2.imwrite("Squares\\"+file + ".png", newimg)	
		x+=1