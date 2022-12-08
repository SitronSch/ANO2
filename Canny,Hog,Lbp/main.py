#!/usr/bin/python

from array import array
import sys
import cv2
import numpy as np
import math
import struct
from datetime import datetime
import glob

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, one_c):
    #https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    
    pts = [((float(one_c[0])), float(one_c[1])),
            ((float(one_c[2])), float(one_c[3])),
            ((float(one_c[4])), float(one_c[5])),
            ((float(one_c[6])), float(one_c[7]))]
    
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(np.array(pts))
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
	    [0, 0],
	    [maxWidth - 1, 0],
	    [maxWidth - 1, maxHeight - 1],
	    [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

    
def main(argv):

    resizeDim=(40,40)
    train_images_full=[(cv2.medianBlur(cv2.imread(img,0),7))   for img in glob.glob("../train_images/full/*.png")]
    train_images_free=[(cv2.medianBlur(cv2.imread(img,0),7))   for img in glob.glob("../train_images/free/*.png")]

    hogData=[]
    for im in train_images_full:
        hogData.append( GetHogVector(cv2.resize(cv2.Canny(im,100,200),resizeDim,interpolation=cv2.INTER_AREA)))
    for im in train_images_free:
        hogData.append(GetHogVector(cv2.resize(cv2.Canny(im,100,200),resizeDim,interpolation=cv2.INTER_AREA)))
        

    labels_full=[1]*len(train_images_full)
    labels_free=[0]*len(train_images_free)

    train_imgs=np.array(train_images_full+train_images_free)
    labels=np.array(labels_full+labels_free)

    lbp=cv2.face.LBPHFaceRecognizer_create()
    lbp.setGridX(4)
    lbp.setGridY(4)
    lbp.train(train_imgs,labels)

  
    svm=TrainSVMWithHog(hogData,labels)

    print('train_images_full')


    pkm_file = open('../train_images/parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []
   
    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)
    
      
    test_images = [img for img in glob.glob("../train_images/test_images/*.jpg")]
    test_images_results=[img for img in glob.glob("../train_images/test_images/*.txt")]
    test_images.sort()
    test_images_results.sort()
    print(pkm_coordinates)
    print("********************************************************")   

    cv2.namedWindow("image",0)
    resultIndex=0
    rightGuess=0
    wrongGuess=0
    image_paint=None
    
    limitTo=-1
    tmpIt=0

    for img_name in test_images:
        tmpIt+=1
        if(tmpIt>=limitTo and limitTo!=-1):
            break
        print(img_name)
        image=cv2.imread(img_name,0)
        image_paint=image.copy()
        fileResult=open(test_images_results[resultIndex],'r')
        currentResults=fileResult.readlines()

        resultIndex+=1

        secondIndexResult=-1
        for coord in pkm_coordinates:
            print("Coord",coord)
            one_place=four_point_transform(image,coord)
            one_place_blur=cv2.medianBlur(one_place,7)
            canny_img=cv2.Canny(one_place_blur,100,200)
            pt_1=(int(coord[0]),int(coord[1]))
            secondIndexResult+=1

            #-------------------------------------------------------------------------------------CANNY tresh
          #  height, width = canny_img.shape
          #  counter=0
          #  for i in range(0,height):
          #      for j in range(0,width):
          #          if(canny_img[i,j]==255):
          #              counter+=1
          #      
          #
          #  if(counter>(height*width)*0.01):      
          #      cv2.circle(image_paint,pt_1,20,(255),-1)
          #      if(currentResults[secondIndexResult]=='1\n'):
          #          rightGuess+=1
          #      else:
          #          wrongGuess+=1
          #  else:
          #      cv2.circle(image_paint,pt_1,20,(100),-1)
          #      if(currentResults[secondIndexResult]=='0\n'):
          #          rightGuess+=1
          #      else:
          #          wrongGuess+=1
#
          #  
#
          #  cell_size = (16, 16)  # h x w in pixels
          #  block_size = (2, 2)  # h x w in cells
          #  nbins = 9  # number of orientation bins
            #-------------------------------------------------------------------------------------------HOG
            
            resized= cv2.resize(canny_img,resizeDim,interpolation=cv2.INTER_AREA)
            resultDataHog=svm.predict(np.matrix(GetHogVector(resized),dtype=np.float32))
#
            if(resultDataHog[1]==1):
                cv2.circle(image_paint,pt_1,20,(255),-1)
                if(currentResults[secondIndexResult]=='1\n'):
                    rightGuess+=1
                else:
                    wrongGuess+=1
            else:
                cv2.circle(image_paint,pt_1,20,(100),-1)
                if(currentResults[secondIndexResult]=='0\n'):
                    rightGuess+=1
                else:
                    wrongGuess+=1

            #-------------------------------------------------------------------------------------------- LBP
           # resized= cv2.resize(one_place_blur,resizeDim,interpolation=cv2.INTER_AREA)
           # label,confidence= lbp.predict(resized)
           # 
           # print(confidence)
           # if(label==1):
           #     cv2.circle(image_paint,pt_1,20,(255),-1)
           #     if(currentResults[secondIndexResult]=='1\n'):
           #         rightGuess+=1
           #     else:
           #         wrongGuess+=1
           # else:
           #     cv2.circle(image_paint,pt_1,20,(100),-1)
           #     if(currentResults[secondIndexResult]=='0\n'):
           #         rightGuess+=1
           #     else:
           #         wrongGuess+=1
         

            
            

        percent=rightGuess/(rightGuess+wrongGuess)*100
        print("{0} % overall success ({1} right, {2} wrong)".format(percent,rightGuess,wrongGuess))
        cv2.imshow("image",image_paint)
        #cv2.imshow("PlaceResize",resized)
        cv2.waitKey(0) 


def TrainSVMWithHog(samples,labels):
    
    samples = np.matrix(samples,dtype=np.float32)
    samples=np.array(samples).astype('float32')
    labels = np.array(labels)
    rand = np.random.RandomState(321)

    shuffle = rand.permutation(len(samples))
    samples = samples[shuffle]
    labels = labels[shuffle]    

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF) # cv2.ml.SVM_LINEAR
    svm.setGamma(5.383)
    svm.setC(2.67)


    svm.train(samples, cv2.ml.ROW_SAMPLE, labels)
    return svm
    
def GetHogVector(image):
    cell_size = (16, 16)  # h x w in pixels
    block_size = (2, 2)  # h x w in cells
    nbins = 9  # number of orientation bins

    dim=image.shape
    
    hog = cv2.HOGDescriptor(_winSize=(dim[0] // cell_size[1] * cell_size[1],dim[1] // cell_size[0] * cell_size[0]),
                    _blockSize=(block_size[1] * cell_size[1],block_size[0] * cell_size[0]),
                    _blockStride=(cell_size[1], cell_size[0]),
                    _cellSize=(cell_size[1], cell_size[0]),
                    _nbins=nbins)
    hogResult=hog.compute(image)

    return hogResult
      
if __name__ == "__main__":
   main(sys.argv[1:])     
