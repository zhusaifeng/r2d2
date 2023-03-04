import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform
import plotmatch
from skimage import measure
from skimage import transform
plt.rcParams['font.sans-serif'] = ['SimHei']

def get_model_params(image1,image2):
    des_left=feat1['descriptors']
    des_right=feat2['descriptors']
    kps_left=feat1['keypoints']
    kps_right=feat2['keypoints']
    # Flann特征匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=40)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_left, des_right, k=2)

    goodMatch = []
    locations_1_to_use = []
    locations_2_to_use = []

    # 匹配对筛选
    min_dist = 1000
    max_dist = 0
    disdif_avg = 0
    # 统计平均距离差
    for m, n in matches:
        disdif_avg += n.distance - m.distance
    disdif_avg = disdif_avg / len(matches)

    for m, n in matches:
        # 自适应阈值
        if n.distance > m.distance + disdif_avg:
            goodMatch.append(m)
            p2 = cv2.KeyPoint(kps_right[m.trainIdx][0], kps_right[m.trainIdx][1], 1)
            p1 = cv2.KeyPoint(kps_left[m.queryIdx][0], kps_left[m.queryIdx][1], 1)
            locations_1_to_use.append([p1.pt[0], p1.pt[1]])
            locations_2_to_use.append([p2.pt[0], p2.pt[1]])
    # goodMatch = sorted(goodMatch, key=lambda x: x.distance)
    print('match num is %d' % len(goodMatch))
    locations_1_to_use = np.array(locations_1_to_use)
    locations_2_to_use = np.array(locations_2_to_use)
    # Perform geometric verification using RANSAC.
    _RESIDUAL_THRESHOLD = 30
    model, inliers = measure.ransac((locations_1_to_use, locations_2_to_use),
                                    transform.AffineTransform,
                                    min_samples=3,
                                    residual_threshold=_RESIDUAL_THRESHOLD,
                                    max_trials=1500)
    print('scale:', model.scale)
    print('Found %d inliers' % sum(inliers))
    # 绘制提取的特征点
    # draw_img = image2.copy()
    # plotmatch.plot_feature_point(draw_img, locations_2_to_use)
    inlier_idxs = np.nonzero(inliers)[0]
    # #1 绘制匹配连线
    plt.rcParams['savefig.dpi'] = 100  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    plt.rcParams['figure.figsize'] = (4.0, 3.0)  # 设置figure_size尺寸设置figure_size尺寸dpi=150
    _, ax = plt.subplots()
    plotmatch.plot_matches(
        ax,
        image1,
        image2,
        locations_1_to_use,
        locations_2_to_use,
        np.column_stack((inlier_idxs, inlier_idxs)),
        plot_matche_points=False,
        matchline=True,
        matchlinewidth=0.3)
    ax.axis('off')
    ax.set_title('')
    plt.show()
    return model

def show_matching_img(image1, image2, params):
    '''展示获取转换矩阵后得融合图

    Parameters
    ----------
    image1：图1
    image2：图2
    params：配准参数/仿射矩阵AffineTransform

    Returns：融合图
    -------

    '''
    image_output = cv2.warpPerspective(image1, params, (image2.shape[1], image2.shape[0]))
    match_img = cv2.addWeighted(image_output, 0.5, image2, 0.5, 0)
    plt.axis('off')
    plt_show(match_img)

def plt_show(img):
    if img.ndim == 2:
        plt.imshow(img, cmap = 'gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == '__main__':
    image1 = cv2.imread("imgs/part8/1.tif")
    print(image1.shape)
    image2 = cv2.imread("imgs/part8/2.tif")
    image2 = cv2.resize(image2, (640, 512))

    print(image2.shape)
    feat1 = np.load("imgs/part8/1.tif.r2d2")
    feat2 = np.load("imgs/part8/2.tif.r2d2")
    print("Number of keypoints:", len(feat1["keypoints"]))

    model = get_model_params(image1, image2)
    show_matching_img(image1, image2, model.params)


