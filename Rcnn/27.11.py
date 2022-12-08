from random import shuffle
import torch
import torchvision
import torchvision.transforms as transforms
import cv2 as cv2
from dataset import CustomImageDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import glob
from torchvision.io import read_image
from torchsummary import summary
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image

def main():
    PATH = './cifar_net2.pth'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet18=models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    coco_names = [ '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'diningtable', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush' ] 

    for name,child in resnet18.named_children():
        print("name",name)
    for param in resnet18.parameters():
        param.requires_grad_=False
        print("name",name)

    #num_ftrs=resnet18.fc.in_features
    #print("resnet18",resnet18.fc)
    #resnet18.fc=nn.Linear(num_ftrs,2)

    for name,param in resnet18.named_parameters():
        print("\t",name)
        if(("fc" in name) or ("layer4" in name)):
            param.requires_grad=True
    resnet18=resnet18.to(device)

    params_to_update=[param for param in resnet18.parameters() if param.requires_grad==True]
    print(len(params_to_update))

    #resnet18.load_state_dict(torch.load(PATH))               #---------------------------                   
    batch_size = 8

    transform = transforms.Compose(
            [
            transforms.ToTensor(),
            ])

    batch_size = 8

    train_images_full=[(cv2.imread(img,1))   for img in glob.glob("../train_images/full/*.png")]
    train_images_free=[(cv2.imread(img,1))   for img in glob.glob("../train_images/free/*.png")]

    labels_full=[1]*len(train_images_full)
    labels_free=[0]*len(train_images_free)

    train_imgs=np.array(train_images_full+train_images_free)

    labels=np.array(labels_full+labels_free,dtype=np.int64)

    customDataset=CustomImageDataset(_labelsArr=labels,_images=train_imgs, transform=transform)
    customLoader=torch.utils.data.DataLoader(customDataset,batch_size=batch_size,shuffle=True)

    test_images = [img for img in glob.glob("../train_images/test_images/*.jpg")]
    test_images_results=[img for img in glob.glob("../train_images/test_images/*.txt")]
    test_images.sort()
    test_images_results.sort()

    pkm_file = open('../train_images/parking_map_python.txt', 'r')
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
    dropoutUse=False
    dropOut=torch.nn.Dropout(dropoutValue)

    resultIndex=0
    overallGuess=0
    correctGuess=0
        
    resnet18.eval().to(device)  
    for img_name in test_images:
        one_img=cv2.imread(img_name,1)
        one_image_paint=one_img.copy()

        #fileResult=open(test_images_results[resultIndex],'r')
        one_img_rgb=cv2.cvtColor(one_img,cv2.COLOR_BGR2RGB)
        img_pil=Image.fromarray(one_img_rgb)
        imageRCNN=transform(img_pil).to(device)
        imageRCNN=imageRCNN.unsqueeze(0)
        outputsRCNN=resnet18(imageRCNN)
        
        pred_classes=[coco_names[i] for i in outputsRCNN[0]['labels'].cpu().numpy()]
        pred_scores=outputsRCNN[0]['scores'].detach().cpu().numpy()
        pred_bboxes=outputsRCNN[0]['boxes'].detach().cpu().numpy()

        tmpIterator=0
        for boxArr in pred_bboxes:
            predictionScore=pred_scores[tmpIterator]
            if(predictionScore>0.5):
                point1=(int(boxArr[0]),int(boxArr[1]))
                point2=(int(boxArr[2]),int(boxArr[3]))
                cv2.rectangle(one_image_paint,point1,point2,(180,180,180),2)
                text=pred_classes[tmpIterator]+" "+f'{predictionScore:.2f}'
                cv2.putText(one_image_paint,text,point1,cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
            
            tmpIterator+=1


        resultIndex+=1

        cv2.imshow("data",one_image_paint)
        cv2.waitKey()

    #print("Overall accuracy",correctGuess/overallGuess)
    #dataiter = iter(testloader)
    #images, labels = next(dataiter)
    resnet18.train().to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(params_to_update,lr=0.001,momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(customLoader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            if(dropoutUse):
                outputs=resnet18(dropOut(inputs))                #With dropout
            else:
                outputs=resnet18(inputs) 

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 0:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    torch.save(resnet18.state_dict(), PATH)


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