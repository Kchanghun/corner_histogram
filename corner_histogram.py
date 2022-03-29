import cv2
import numpy as np


# img = cv2.imread('./data/putTogether_Color.jpg')
img = cv2.imread('./data/putTogether_Gray.jpg')
# (x, y, )
roi_1st = []
roi_2nd = []

def onMouse(event, x, y, flags, param):
    global roi_1st, roi_2nd
    if event == cv2.EVENT_LBUTTONDOWN:
        #left down --> 1st image
        roi_1st.append([x, y])
        # patch size is 9
        cv2.rectangle(param[0], (x - 4, y - 4),
                                (x + 5, y + 5), (0, 0, 255), 2)    
        # print('roi_1st ',roi_1st)
        # print('roi_2nd ',roi_2nd)
    elif event == cv2.EVENT_RBUTTONDOWN:
        #left down --> 1st image
        roi_2nd.append([x, y])
        # patch size is 9
        cv2.rectangle(param[0], (x - 4, y - 4),
                                (x + 5, y + 5), (0, 255, 0), 2)
        # print('roi_1st ',roi_1st)
        # print('roi_2nd ',roi_2nd)
    
    cv2.imshow('Click center of ROI',param[0])
            

cv2.imshow('Click center of ROI',img)
cv2.setMouseCallback('Click center of ROI',onMouse,[img])
cv2.waitKey()
cv2.destroyAllWindows()