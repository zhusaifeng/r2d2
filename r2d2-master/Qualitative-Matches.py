import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform
import plotmatch
plt.rcParams['font.sans-serif'] = ['SimHei']
pair_idx = 3
assert(pair_idx in [1, 2, 3])
pair_path = os.path.join('images', 'pair_%d' % pair_idx)
# image1 = np.array(Image.open(os.path.join(pair_path, '1.jpg')))
# image2 = np.array(Image.open(os.path.join(pair_path, '2.jpg')))
image1=cv2.imread("imgs/part2/1.jpg")
print(image1.shape)
image2=cv2.imread("imgs/part2/2.jpg")
image2=cv2.resize(image2,(640,512))

print(image2.shape)
feat1 = np.load("imgs/part2/1.jpg.r2d2")
feat2 = np.load("imgs/part2/2.jpg.r2d2")
print("Number of keypoints:",len(feat1["keypoints"]))
matches = match_descriptors(feat1['descriptors'], feat2['descriptors'], cross_check=True)
print('Number of raw matches: %d.' % matches.shape[0])
keypoints_left = feat1['keypoints'][matches[:, 0], : 2]
keypoints_right = feat2['keypoints'][matches[:, 1], : 2]
np.random.seed(0)
model, inliers = ransac(
    (keypoints_left, keypoints_right),
    ProjectiveTransform, min_samples=4,
    residual_threshold=4, max_trials=10000
)
inlier_idxs = np.nonzero(inliers)[0]
n_inliers = np.sum(inliers)
print('Number of inliers: %d.' % n_inliers)
inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_left[inliers]]
inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_right[inliers]]
placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
# inlier_keypoints_left=np.array(inlier_keypoints_left)
# inlier_keypoints_right=np.array(inlier_keypoints_right)
# #1 绘制匹配连线
plt.rcParams['savefig.dpi'] = 100  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
plt.rcParams['figure.figsize'] = (4.0, 3.0)  # 设置figure_size尺寸设置figure_size尺寸dpi=150
_, ax = plt.subplots()
plotmatch.plot_matches(
    ax,
    image1,
    image2,
    keypoints_left,
    keypoints_right,
    np.column_stack((inlier_idxs, inlier_idxs)),
    plot_matche_points=False,
    matchline=True,
    matchlinewidth=0.3)
ax.axis('off')
ax.set_title('')
plt.show()



# image3 = cv2.drawMatches(image1, inlier_keypoints_left, image2, inlier_keypoints_right, placeholder_matches, None)
#
#
#
# plt.imshow(image3[:,:,::-1])
# plt.title("r2d2特征点匹配")
# plt.show()
# cv2.imwrite("pz2.jpg",image3[:,:,::-1])
# 配准
goodMatch= placeholder_matches[:20]
ptsA = np.float32([inlier_keypoints_left[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
ptsB = np.float32([inlier_keypoints_right[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
H, status = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 4)
print(H)
imgOut = cv2.warpPerspective(image1, H, (image2.shape[1], image2.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
cv2.imwrite("imgOut.jpg",imgOut)
print(image2.shape,imgOut.shape)
overlapping = cv2.addWeighted(image2, 0.5, imgOut, 0.5, 0)
cv2.imwrite("overlappig.jpg",overlapping)
# cv2.imwrite("pz.jpg",image3)

# 显示对比
plt.subplot(221)
plt.title('待配准图像')
plt.imshow(image1[:,:,::-1])

plt.subplot(222)
plt.title('参考图像')
plt.imshow(image2[:,:,::-1])


plt.subplot(223)
plt.title('仿射变换输出图像')
plt.imshow(imgOut[:,:,::-1])

plt.subplot(224)
plt.title('配准结果')
plt.imshow(overlapping[:,:,::-1])

plt.show()
