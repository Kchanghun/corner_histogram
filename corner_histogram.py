from turtle import distance
import cv2
import numpy as np
from matplotlib import pyplot as plt

#img = cv2.imread('./data/putTogether_Color.jpg')
img = cv2.imread('./data/putTogether_Gray.jpg',cv2.IMREAD_GRAYSCALE)
font=cv2.FONT_HERSHEY_SIMPLEX

# top left of ROI [x, y]
roi_1st_anchor = []
roi_2nd_anchor = []

# Roi Image
roi_1st_IM = []
roi_2nd_IM = []

patch_size = 9

compareHist_result = []

def onMouse(event, x, y, flags, param):
    global roi_1st_anchor, roi_2nd_anchor, patch_size
    if event == cv2.EVENT_LBUTTONDOWN:
       if flags & cv2.EVENT_FLAG_SHIFTKEY:
           # left down with Shiftkey
           for i in range(4):
                getRoiImage(1, roi_1st_anchor[i])
                getRoiImage(2, roi_2nd_anchor[i])
       else:
            # left down only --> 1st image
            roi_1st_anchor.append([y, x])
            index = roi_1st_anchor.__len__()
            # red color patch size is 9
            cv2.putText(img,str(index),[x,y],font,1,255,2)
            cv2.rectangle(param[0], (x - patch_size // 2, y - patch_size // 2),
                          (x + patch_size // 2, y + patch_size // 2), (0, 0, 255), 2)
            # print('roi_1st_anchor ',roi_1st_anchor)
    elif event == cv2.EVENT_RBUTTONDOWN:
        # right down --> 2nd image
        roi_2nd_anchor.append([y, x])
        index = roi_2nd_anchor.__len__()
        # green color patch size is 9
        cv2.putText(img,str(index),[x,y],font,1,255,2)
        cv2.rectangle(param[0], (x - patch_size // 2, y - patch_size // 2),
                      (x + patch_size // 2, y + patch_size // 2), (0, 255, 0), 2)
        # print('roi_2nd_anchor ',roi_2nd_anchor)
    
    cv2.imshow('Match ROI',param[0])

def getRoiImage(which, anchor):
    global img, roi_1st_IM, roi_2nd_IM, patch_size
    if which == 1:
        roi_1st_IM.append(img[anchor[0]-patch_size//2:anchor[0]+patch_size//2,
                              anchor[1]-patch_size//2:anchor[1]+patch_size]//2)
        # cv2.imshow('1st Rois '+str(roi_1st_IM.__len__()),roi_1st_IM[roi_1st_IM.__len__()-1])
    elif which == 2:
        roi_2nd_IM.append(img[anchor[0]-patch_size//2:anchor[0]+patch_size//2,
                              anchor[1]-patch_size//2:anchor[1]+patch_size]//2)
         
        # cv2.imshow('2nd Rois '+str(roi_1st_IM.__len__()),roi_1st_IM[roi_1st_IM.__len__()-1])
        
    if roi_2nd_IM.__len__() == 4:
        drawHist()
    
def drawHist():
    global roi_1st_IM, roi_2nd_IM
    roi_1st_hist = []
    roi_2nd_hist = []
    
    for i in range(2):
        for index in range(4):
            histSize=8
            plt.subplot(2,4,i*4+index+1)
            binX = np.arange(histSize)
            if i == 0:
                plt.title('1st - '+str(index))
                roi_1st_hist.append(cv2.calcHist(images=[roi_1st_IM[index]],channels=[0],mask=None,
                                     histSize=[histSize], ranges=[0,256]))
                cv2.normalize(roi_1st_hist[index],roi_1st_hist[index],1,0,cv2.NORM_L1)
                roi_1st_hist[index] = roi_1st_hist[index].flatten()
                plt.bar(binX,roi_1st_hist[index],width=1,color = 'b')
            elif i == 1:
                plt.title('2nd - '+str(index))
                roi_2nd_hist.append(cv2.calcHist(images=[roi_2nd_IM[index]],channels=[0],mask=None,
                                     histSize=[histSize], ranges=[0,256]))
                cv2.normalize(roi_2nd_hist[index],roi_2nd_hist[index],1,0,cv2.NORM_L1)
                roi_2nd_hist[index] = roi_2nd_hist[index].flatten()
                plt.bar(binX,roi_2nd_hist[index],width=1,color = 'r')
    
    distanceCompare(roi_1st_hist,roi_2nd_hist)
    plt.tight_layout()
    plt.show()

def distanceCompare(roi_1st_hist,roi_2nd_hist):
    global compareHist_result
    distance = []
    for i in range(4):
        for j in range(4):
            distance.append([np.absolute(
                cv2.compareHist(
                roi_1st_hist[i],
                roi_2nd_hist[j],
                cv2.HISTCMP_CORREL
            )),i,j])
    
    distance.sort(reverse=True)
    compareHist_result = distance[:4]    
    


cv2.imshow('Match ROI',img)
cv2.setMouseCallback('Match ROI',onMouse,[img])
cv2.waitKey()
cv2.destroyAllWindows()