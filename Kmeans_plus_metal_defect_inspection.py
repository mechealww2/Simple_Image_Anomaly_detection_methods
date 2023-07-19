import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import random
import time
import math

#######################################################################################################################
# 计算聚类中心辅助函数
def distance_func(point1, point2):
    distance = np.linalg.norm(point1 - point2)
    return distance

def nearest(point, cluster_centers):
    '''
    计算point和cluster_centers之间的最小距离
    :param point: 当前的样本点
    :param cluster_centers: 当前已经初始化的聚类中心
    :return: 返回point与当前聚类中心的最短距离
    '''
    FLOAT_MAX = math.e-2
    min_dist = FLOAT_MAX
    m = np.shape(cluster_centers)[0]  # 当前已经初始化聚类中心的个数
    for i in range(m):
        # 计算point与每个聚类中心之间的距离
        d = distance_func(point, cluster_centers[i, ])
        # 选择最短距离
        if min_dist > d:
            min_dist = d
    return min_dist

# 计算聚类中心的主函数
def get_cent(points, k):
    '''
    kmeans++的初始化聚类中心的方法
    :param points: 样本
    :param k: 聚类中心的个数
    :return: 初始化后的聚类中心
    '''
    m, n = np.shape(points)
    cluster_centers = np.mat(np.zeros((k, n)))

    # 1、随机选择一个样本点作为第一个聚类中心
    index = np.random.randint(0, m)
    cluster_centers[0,] = np.copy(points[index,])  # 复制函数，修改cluster_centers，不会影响points

    # 2、初始化一个距离序列
    d = [0.0 for _ in range(m)]

    for i in range(1, k):
        sum_all = 0
        for j in range(m):
            # 3、对每一个样本找到最近的聚类中心点
            d[j] = nearest(points[j,], cluster_centers[0:i, ])
            # 4、将所有的最短距离相加
            sum_all += d[j]
        # 5、取得sum_all之间的随机值
        sum_all *= random.random()
        # 6、获得距离最远的样本点作为聚类中心点
        for j, di in enumerate(d):  # enumerate()函数用于将一个可遍历的数据对象（如列表、元组或字符串）组合为一个索引序列，同事列出数据和数据下标一般用在for循环中
            sum_all -= di
            if sum_all > 0:
                continue
            cluster_centers[i] = np.copy(points[j,])
            break
    return cluster_centers
#######################################################################################################################
if __name__ == '__main__':
    # Initialize rgb pixel values for each class in kmeans using specific values
    # Generate 10 equidistant colors
    # Define a list of 12 colors
    bgr_list = [(0, 0, 255),
                 (0, 255, 0),
                 (255, 0, 0),
                 (128, 128, 255),
                 (128, 255, 128),
                 (255, 128, 128),
                 (128, 0, 255),
                 (128, 255, 0),
                 (255, 128, 0),
                 (0, 128, 255),
                 (0, 255, 128),
                 (255, 0, 128)]

    # Create the colormap
    # selec_bgr = ListedColormap(colors)
    # bgr_list = selec_bgr.colors
    # Reading images using matplotlib library
    # image = mpimg.imread('./data/img_sharp.png')
    image = mpimg.imread('./data/Tianchi_metal_defect.png')
    height, width, channel = image.shape
    # show original image
    plt.figure()
    plt.subplot(3, 3, 1)
    plt.axis('off')
    plt.title('Original')
    plt.imshow(image)

    # do kmeans segmentation
    for i, k in enumerate(range(5, 13, 1)):
        # 计时
        str_time = time.time()
        # extract bgr features
        features = []
        print(f"正在计算k={k},第{k-4}张聚类图...")
        # try:
        for y in range(height):
            for x in range(width):
                features.append(image[y, x, :] / 255)
        features = np.array(features)
        # initial segments center using random value in features
        # kmeans_centers = features[np.random.choice(len(features), k), :]
        kmeans_centers = get_cent(features, k)
        kmeans_centers = np.array(kmeans_centers)
        # except:
        #     pass
        # update
        while True:
            # calculate distance matrix
            def euclidean_dist(X, Y):
                Gx = np.matmul(X, X.T)
                Gy = np.matmul(Y, Y.T)
                diag_Gx = np.reshape(np.diag(Gx), (-1, 1))
                diag_Gy = np.reshape(np.diag(Gy), (-1, 1))
                return diag_Gx + diag_Gy.T - 2 * np.matmul(X, Y.T)

            dist_matrix = []
            for start in range(0, len(features), 1000):
                dist_matrix.append(euclidean_dist(features[start:start + 1000, :], kmeans_centers))
            dist_matrix = np.concatenate(dist_matrix, axis=0)
            # dist_matrix = euclidean_dist(features, kmeans_centers)
            # get seg class for each sample
            segs = np.argmin(dist_matrix, axis=1)
            # update new kmeans center
            new_kmeans_centers = []
            for j in range(k):
                new_kmeans_centers.append(np.mean(features[segs == j, :], axis=0))
            new_kmeans_centers = np.array(new_kmeans_centers)
            # calculate whether converge
            if np.mean(abs(kmeans_centers - new_kmeans_centers)) < 0.1:
                break
            else:
                kmeans_centers = new_kmeans_centers
        # assign
        segs = segs.reshape(height, width)
        seg_result = np.zeros((height, width, channel), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                seg_result[y, x, :] = bgr_list[segs[y, x]]

        sto_time = time.time()
        print(f"第{k-4}次聚类计算的时间消耗为：{round(sto_time-str_time,3)}s")
        # show kmeans result
        plt.subplot(3, 3, i + 2)
        plt.title('k={}'.format(k))
        plt.axis('off')
        plt.imshow(seg_result)
        plt.savefig('Tianchi_metal_defect_kmeans_plus111.jpg')
