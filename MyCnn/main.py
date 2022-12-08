from random import shuffle
import torch
import torchvision
import torchvision.transforms as transforms
import cv2 as cv2
from cnn import CNN
from dataset import CustomImageDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import glob
from torchvision.io import read_image
from torchsummary import summary
import matplotlib.pyplot as plt

def main():
    transform = transforms.Compose(
        [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    batch_size = 8
    PATH = './cifar_netAvg.pth'

    train_images_full=[(cv2.imread(img,1))   for img in glob.glob("E:/OneDrive/OneDrive - VSB-TUO/ANO2/Handin/train_images/full/*.png")]
    train_images_free=[(cv2.imread(img,1))   for img in glob.glob("E:/OneDrive/OneDrive - VSB-TUO/ANO2/Handin/train_images/free/*.png")]

    labels_full=[1]*len(train_images_full)
    labels_free=[0]*len(train_images_free)

    train_imgs=np.array(train_images_full+train_images_free)

    labels=np.array(labels_full+labels_free,dtype=np.int64)
    
    customDataset=CustomImageDataset(_labelsArr=labels,_images=train_imgs, transform=transform)
    customLoader=torch.utils.data.DataLoader(customDataset,batch_size=batch_size,shuffle=True)


    test_images = [img for img in glob.glob("E:/OneDrive/OneDrive - VSB-TUO/ANO2/Handin/train_images/test_images/*.jpg")]
    test_images_results=[img for img in glob.glob("E:/OneDrive/OneDrive - VSB-TUO/ANO2/Handin/train_images/test_images/*.txt")]
    test_images.sort()
    test_images_results.sort()

    pkm_file = open('E:/OneDrive/OneDrive - VSB-TUO/ANO2/Handin/train_images/parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []

    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)

    classes=('NOCAR','CAR')

    dataset_size=len(customLoader.dataset)
    print("dataset_size",dataset_size)

    dropoutValue=0.5
    dropoutUse=True
    dropOut=torch.nn.Dropout(dropoutValue)

    net=CNN()
    net.load_state_dict(torch.load(PATH))
    resultIndex=0
    overallGuess=0
    correctGuess=0
    net.eval()
    
    for img_name in test_images:
        image=cv2.imread(img_name,1)
        image_paint=image.copy()

        fileResult=open(test_images_results[resultIndex],'r')
        currentResults=fileResult.readlines()
        secondIndexResult=0
        for coord in pkm_coordinates:
            #print("Coord",coord)
            one_place=four_point_transform(image,coord)
            one_place=cv2.resize(one_place,(80,80),interpolation=cv2.INTER_AREA)

            tensor=transform(one_place)
            tensor=tensor.unsqueeze(0)

            if(dropoutUse):
                outputs=net(dropOut(tensor))                #With dropout
            else:
                outputs=net(tensor)                        #Without dropout

            _,predicted=torch.max(outputs,1)
            pt_1=(int(coord[0]),int(coord[1]))
            if(predicted[0]==1):                
                cv2.circle(image_paint,pt_1,20,(255,255,255),-1)
                if(currentResults[secondIndexResult]=='1\n'):
                    correctGuess+=1
            else:
                cv2.circle(image_paint,pt_1,20,(100,100,100),-1)
                if(currentResults[secondIndexResult]=='0\n'):
                    correctGuess+=1
            secondIndexResult+=1
            overallGuess+=1

        resultIndex+=1
        cv2.imshow("data",image_paint)
        cv2.waitKey()

    print("Overall accuracy",correctGuess/overallGuess)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(customLoader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            if(dropoutUse):
                outputs=net(dropOut(inputs))                #With dropout
            else:
                outputs=net(inputs)                        #Without dropout

            

            #outputs = net(tensor)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 0:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    torch.save(net.state_dict(), PATH)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    cv2.imshow("img",img.numpy().transpose(1,2,0))
    cv2.waitKey(2)

def fx(x):
    return x**2-6*x+1

def deriv(x):
    return 2*x-6

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

if __name__ == "__main__":
   main()     