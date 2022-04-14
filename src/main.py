import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os


def PCA(X,reduce_to_dim):
    """

    :param X: 输入数据(n,m)
    :param reduce_to_dim: 要降低到的维度
    :return: W,mu 线性变换阵，以及数据规范化前均值
    """

    size = X.shape[0]
    mu = np.array([1 / size * np.sum(X, axis=0)])
    var_x = 1 / (size - 1) * np.sum((X - mu)*(X - mu), axis=0)  # 样本
    X = (X - mu)  # 标准化
    R = 1 / (size - 1) * np.dot(X.T, X)  # (m,m)相关矩阵
    values, feat_vectors = np.linalg.eig(R)
    sorted_indices = np.argsort(values)
    a = sorted_indices[:-reduce_to_dim-1:-1]
    W = feat_vectors[:,a]
    return W,mu,var_x**(1/2)


def generate_data(data_dimension = 3, number=500):
    """ 生成3维数据 """

    mean = [1, 2, 3]
    cov = [[0.01, 0, 0], [0, 1, 0], [0, 0, 2]]
    sample_data = []
    for index in range(number):
        sample_data.append(np.random.multivariate_normal(mean, cov).tolist())
    X = np.array(sample_data)
    X[:,1:3] = X[:,1:3].dot(np.array([[3**(1/2)/2,-1./2],[1./2,3**(1/2)/2]]))  # 对数据进行旋转
    return X


def draw_data(dimension, origin_data, pca_data):
    """ 将PCA前后的数据进行可视化对比 """

    # 绘制3维框图
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(origin_data[:, 0], origin_data[:, 1], origin_data[:, 2],
                label='Origin Data')
    ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], color='r', label='PCA Data')
    # ax.zaxis.set_major_locator(plt.MultipleLocator(1))
    # # ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    # ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    # plt.xlim(-5,5)
    # plt.ylim(-5,5)
    plt.title("PCA Model")
    plt.legend()
    plt.show()

def hand_gene_test():
    """
    对人工生成数据进行PCA降维，并展示
    :return:
    """
    data = generate_data(3)
    W, mu, standard_div =PCA(data,2)  # 返回由训练集生成的的变换矩阵，以及训练集的均值、标准差
    print("生成的线性变换矩阵为：")
    print(W)
    pca_data = np.dot(data - mu,W)
    refac_data = np.dot(pca_data,W.T) + mu  # 对降维的数据进行重构
    draw_data(3, data, refac_data)
    return

def read_fac_image(file_path):
    face_list = os.listdir(file_path)  # 从该目录中读取文件名列表
    data = []
    for face_file in face_list:  # 处理每一张人脸图片
        this_path = os.path.join(file_path, face_file)
        with open(this_path) as f:
            face_img = cv2.imread(this_path)  # 利用opencv读取图片
            size = (40,40)  # 压缩图片 加快处理速度
            face_img = cv2.resize(face_img, size)
            row,col,z = face_img.shape
            face_img = face_img.reshape(row * col * z)
            data.append(face_img)
    data = np.array(data)
    return data


def asian_face_pca(reduce_to):
    """
    读取人脸数据集
    :param reduce_to: 希望降低到的维度
    :return:
    """

    data = read_fac_image("/Users/VitoLiu/Downloads/asia")  # 从该文件夹读取人脸照片 共含有10张人脸照片
    W, mu, standard_div = PCA(data, reduce_to)  # 返回由数据集生成的的变换矩阵，以及训练集的均值、标准差
    W = np.real(W)  # 如果产生复数均转化为实数
    pca_data = np.dot(data - mu, W)  # 降维后的数据
    refac_data = np.dot(pca_data, W.T) + mu  # 对降维的数据进行重构
    refac_data = refac_data.astype(int)

    # 显示每一张降维后的人脸照片
    j = 0
    for i in range(refac_data.shape[0]):
        psnr = calc_psnr(data[i],refac_data[i])
        print("第"+ str(i+1)+"张图片的信噪比为：",end='')
        print(psnr)
        if j < 2:
            plt.imshow(refac_data[i].reshape(40,40,3))
            plt.title("PCA picture "+ str(j +1))
            plt.show()
            j += 1
    return

def calc_psnr(formal_img, pca_img):
    mse = np.mean((formal_img - pca_img) ** 2)
    psnr = 20 * np.log10(255 / (mse)**(1/2))
    return psnr


# hand_gene_test()  # 观察手动生成数据集的效果
asian_face_pca(10)  # 观察人脸数据集的效果
