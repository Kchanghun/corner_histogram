import cv2
from matplotlib import pyplot as plt

path = './data/'
img1st = cv2.imread(path + '1st.jpg')
img2nd = cv2.imread(path + '2nd.jpg')
# img1st = cv2.imread(path+'1st.jpg',cv2.IMREAD_GRAYSCALE)
# img2nd = cv2.imread(path+'2nd.jpg',cv2.IMREAD_GRAYSCALE)

img1st = cv2.cvtColor(img1st, cv2.COLOR_BGR2RGB)
img2nd = cv2.cvtColor(img2nd, cv2.COLOR_BGR2RGB)
# img1st = cv2.cvtColor(img1st,cv2.IMREAD_GRAYSCALE)
# img2nd = cv2.cvtColor(img2nd,cv2.IMREAD_GRAYSCALE)


# put img2 next to img1
fig, ax = plt.subplots(1, 2, figsize=(10,5), sharey=True)
# print(fig)

ax[0].axis('off')
ax[0].imshow(img1st,aspect='auto')

ax[1].axis('off')
ax[1].imshow(img2nd,aspect='auto')

plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

plt.gcf().canvas.manager.set_window_title('put two images in one canvas')
plt.savefig('./data/putTogether_Color.jpg',bbox_inches='tight')
# plt.savefig('./data/putTogether_Gray.jpg',bbox_inches='tight')
plt.show()

