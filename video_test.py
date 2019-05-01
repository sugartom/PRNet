import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from time import time

from api import PRN
from utils.write import write_obj_with_colors

import cv2
from utils.cv_plot import plot_kpt

from PIL import Image
from subprocess import Popen, PIPE

# ---- init PRN
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # GPU number, -1 for CPU
prn = PRN(is_dlib = True) 
save_folder = 'TestImages/results'


cam = cv2.VideoCapture("./tensorflow_face_detection/media/test.mp4")
frame_width = int(cam.get(3))
frame_height = int(cam.get(4))

ind = 0

fps, duration = 24, 100

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 24, (frame_width,frame_height))

while(True):
	ret_val, img = cam.read()
	if(ret_val):
		image = img

		pos = prn.process(image) # use dlib to detect face

		if pos is not None:

			# -- Basic Applications
			# get landmarks
			kpt = prn.get_landmarks(pos)
			# # 3D vertices
			# vertices = prn.get_vertices(pos)
			# # corresponding colors
			# colors = prn.get_colors(image, vertices)
			# cv2.imwrite(os.path.join(save_folder,  '{}_nn_kpt.jpg'.format(ind)), plot_kpt(image, kpt))

			out.write(plot_kpt(image, kpt))
			print("capture frame count: {}".format(ind))


			# -- save
			# np.savetxt(os.path.join(save_folder, name + '.txt'), kpt) 
			# write_obj_with_colors(os.path.join(save_folder, name + '.obj'), vertices, prn.triangles, colors) #save 3d face(can open with meshlab)

			# sio.savemat(os.path.join(save_folder, name + '_mesh.mat'), {'vertices': vertices, 'colors': colors, 'triangles': prn.triangles})
	# Press Q on keyboard to stop recording
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	ind = ind + 1

	if ind > 1000:
		break

# When everything done, release the video capture and video write objects
cam.release()
out.release()

