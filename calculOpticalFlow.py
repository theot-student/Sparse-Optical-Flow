import sys
import of_l1, of_l2, of_mix
import torch
import cv2
import numpy as np
from skimage.io import imread, imsave
import os
import tiffcapture as tc
from time import time

#function to convert RGB images to Grayscale
def convertRGB2GRAY(inputImage):
    outputImage = 0.299 * inputImage[...,0] + 0.587 * inputImage[...,1] + 0.114 * inputImage[...,0]
    return outputImage.astype(np.float32)

#np.set_printoptions(threshold=sys.maxsize)

class OtpicalFlowSparse():

    def __init__(self, filename = "VehicleTrafficking.tif",     #name of video
                path = "videos/VehicleTrafficking.tif" ,                        #path of directory of video
                weights = [0.5] ,                       #model weight between TV-L1 and sparsity. Value is in  [0, 1]. 0 = sparse; 1 = not sparse
                regs = [0.85] ,                                      #Regularization weight. Value is in [0, 1]. 0 = very regularized
                gradient_step = 0.01 ,                                 #gradient step for ADAM scheduler
                precision = 1e-7 ,                                 #stop precision for optimization
                isRGB = True ,                                       #booleans for differents color of videos
                isRed = False ,
                nbOfImages = 1  ,                                     #number of images pair to process, if 0 process all video
                save_directory = "save/cometes/",
                method = 1,                                          #method used in term of norm. 1 = l_1 ; 2 = l_2 ; 3 = mix l_1 l_2
                init = 0):                                          #initialization for the motion field. 0 = init at zero ; 1 = init at random between -1 and 1 ; 2 = init at random between -0.1 and 0.1

        self.filename = filename
        self.path = path
        self.weights = weights
        self.regs = regs
        self.gradient_step = gradient_step
        self.precision = precision
        self.isRGB = isRGB
        self.isRed = isRed
        self.nbOfImages = nbOfImages
        self.save_directory = save_directory
        self.method = method
        self.init = init

    def calculOpticalFlow(self):
        #read the video
        cap = []
        ret, cap = cv2.imreadmulti(mats=cap,
                                      filename=self.path,
                                      flags=cv2.IMREAD_ANYCOLOR)

        #check number of images
        if self.nbOfImages == 0:
            nbOfImages = np.shape(cap)[0] - 1
        else:
            nbOfImages = min(np.shape(cap)[0] - 1, self.nbOfImages)


        #compute the motion fields for every parameters
        for reg in self.regs:
            for weight in self.weights:

                save_path = self.save_directory + str(weight) + "_" + str(reg) + "_" + str(self.precision) + "_" + self.filename

                old_frame = cap[0]

                if self.isRGB:
                    grayFrame = convertRGB2GRAY(old_frame)
                elif self.isRed:
                    grayFrame = old_frame[...,2]
                else :
                    grayFrame = old_frame

                old_frame_torch = torch.from_numpy(grayFrame)

                # create HSV to store motion field color representation
                hsvShape = (np.shape(old_frame)[0], np.shape(old_frame)[1], 3)


                motion_fieldShape = (nbOfImages,np.shape(old_frame)[0], np.shape(old_frame)[1], 3)

                hsv = np.zeros(hsvShape, np.uint8)
                hsv[..., 2] = 255


                motion_field = np.ndarray(motion_fieldShape)

                if self.method == 1:
                    opticalFlowGenerator = of_l1.OF(weight, reg, self.gradient_step, self.precision, self.init)
                elif self.method == 2:
                    opticalFlowGenerator = of_l2.OF(weight, reg, self.gradient_step, self.precision, self.init)
                elif self.method == 3:
                    opticalFlowGenerator = of_mix.OF(weight, reg, self.gradient_step, self.precision, self.init)
                else:
                    opticalFlowGenerator = of_l1.OF(weight, reg, self.gradient_step, self.precision, self.init)

                for i in range(nbOfImages):
                    start = time()
                    if self.isRGB:
                        grayNewFrame = convertRGB2GRAY(cap[i+1])
                    elif self.isRed:
                        grayNewFrame = cap[i+1][...,2]
                    else :
                        grayNewFrame = cap[i+1]

                    new_frame_torch = torch.from_numpy(grayNewFrame)

                    #compute motion field optical flow
                    motion_field_temp = opticalFlowGenerator(old_frame_torch, new_frame_torch).detach().numpy()


                    old_frame_torch = new_frame_torch

                    #convert the algorithm's output into Polar coordinates
                    mag, ang = cv2.cartToPolar(motion_field_temp[..., 0], motion_field_temp[..., 1])

                    #normalize magnitude of optical flow for clarity of color representaion
                    magNorm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                    magNorm = 255 * np.ones(np.shape(magNorm)) / (np.ones(np.shape(magNorm)) + np.exp(0.03 * (-magNorm + 127.5 * np.ones(np.shape(magNorm)))))

                    # Use Hue and Saturation to encode the Optical Flow
                    hsv[..., 0] = np.mod(-(ang * 180 / np.pi / 2) - 45, 180)
                    hsv[..., 1] = magNorm

                    """
                    #print distribution of optical flow motion field magnitude to check
                    print(np.max(mag))
                    print(np.min(mag))
                    print(np.median(mag))
                    print(np.mean(mag))
                    """

                    # Convert HSV image into RGB for demo
                    motion_field[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                    stop = time()
                    print("temps image ", i, " : ", stop - start)

                #save the optical flow as video in save directory
                cv2.imwritemulti(save_path, motion_field)

#data
filename = "VehicleTrafficking_DOG.tif"     #name of video
path = "videos/" + filename                         #path of directory of video
weights = [0.9]                        #model weight between TV-L1 and sparsity. Value is in  [0, 1]. 0 = sparse; 1 = not sparse
regs = [0.7]                                       #Regularization weight. Value is in [0, 1]. 0 = very regularized
gradient_step = 0.01                                 #gradient step for ADAM scheduler
precision = 1e-10                                  #stop precision for optimization
isRGB = False                                        #booleans for differents color of videos
isRed = False
nbOfImages = 1                                       #number of images pair to process, if 0 process all video
save_directory = "save/ComparaisonSensibilite/"
method = 1                                         #method used in term of norm. 1 = l_1 ; 2 = l_2 ; 3 = mix l_1 l_2
init = 0                                         #initialization for the motion field. 0 = init at zero ; 1 = init at random between -1 and 1 ; 2 = init at random between -0.1 and 0.1

ofs = OtpicalFlowSparse(filename, path, weights, regs, gradient_step, precision, isRGB, isRed, nbOfImages, save_directory, method, init)
ofs.calculOpticalFlow()
