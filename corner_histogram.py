import cv2
import numpy as np


#img = cv2.imread('./data/putTogether_Color.jpg')
img = cv2.imread('./data/putTogether_Gray.jpg',cv2.IMREAD_GRAYSCALE)

# top left of ROI [x, y]
roi_1st_origin = []
roi_2nd_origin = []

patch_size = 9

# Roi Image
roi_1st_IM = []
roi_2nd_IM = []

def onMouse(event, x, y, flags, param):
    global roi_1st_origin, roi_2nd_origin
    if event == cv2.EVENT_LBUTTONDOWN:
       if flags & cv2.EVENT_FLAG_SHIFTKEY:
           # left down with Shiftkey
           for i in range(4):
                getRoiImage(1, roi_1st_origin[i])
                getRoiImage(2, roi_2nd_origin[i])
       else:
            # left down only --> 1st image
            roi_1st_origin.append([x, y])
            print(roi_1st_origin)
            # red color patch size is 9
            print('l',x-patch_size//2)
            cv2.rectangle(param[0], (x - patch_size // 2, y - patch_size // 2),
                          (x + patch_size // 2, y + patch_size // 2), (0, 0, 255), 2)
            # cv2.imshow('Click center of ROI',param[0])
            # print('roi_1st_origin ',roi_1st_origin)
    elif event == cv2.EVENT_RBUTTONDOWN:
        # right down --> 2nd image
        roi_2nd_origin.append([x, y])
        # green color patch size is 9
        print('r',x-patch_size//2)
        cv2.rectangle(param[0], (x - patch_size // 2, y - patch_size // 2),
                      (x + patch_size // 2, y + patch_size // 2), (0, 255, 0), 2)
        # print('roi_2nd_origin ',roi_2nd_origin)
    
    cv2.imshow('Click center of ROI',param[0])

def getRoiImage(which, origin):
    global img, roi_1st_IM, roi_2nd_IM
    if which == 1:
        roi_1st_IM.append(img[origin[0]:origin[0]+patch_size,
                              origin[1]:origin[1]+patch_size])
        # cv2.imshow('1st Rois '+str(roi_1st_IM.__len__()),roi_1st_IM[roi_1st_IM.__len__()-1])
    elif which == 2:
        roi_2nd_IM.append(img[origin[0]:origin[0]+patch_size,
                              origin[1]:origin[1]+patch_size])
        # cv2.imshow('2nd Rois '+str(roi_1st_IM.__len__()),roi_1st_IM[roi_1st_IM.__len__()-1])
    
    

cv2.imshow('Click center of ROI',img)
cv2.setMouseCallback('Click center of ROI',onMouse,[img])
cv2.waitKey()
cv2.destroyAllWindows()