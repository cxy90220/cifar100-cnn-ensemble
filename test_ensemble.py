import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
from keras.models import load_model

# 原始数据读入成字典
def unpickle(file: str) -> dict:
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# 数据从多维数组转换成原图片shape
def reshape_data(data_set: np.ndarray) -> np.ndarray:
    return data_set.reshape(data_set.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)

# 从数据文件读取出数据和标签
def read_data(file_path: str) -> np.ndarray:
    data_set = {key.decode('utf8'): value for key, value in unpickle(file_path).items()}
    return np.array(data_set['fine_labels']), data_set['data']

def ensemble_prediction(test_data: np.ndarray) -> list:
    bincount = np.bincount
    result = np.array([np.argmax(model.predict(test_data), 1) for model in model_list])
    result = result.transpose(1, 0)
    return [bincount(i).argmax() for i in result]  # 选取投票数最多的

if __name__ == '__main__':
    # 数据文件路径
    test_set_path = 'cifar-100-python/test'

    # 标签数字代表的细标签名称
    fine_label_names = [i.decode('utf8') for i in unpickle('cifar-100-python/meta')[b'fine_label_names']]

    # 读取测试集
    test_label, test_data = read_data(test_set_path)

    # 按原图片的size来reshape多维数组
    test_data = reshape_data(test_data)

    # 图像的RGB值映射到(0,1)范围
    test_data = test_data / 255.0
    # 训练9个模型来集成
    epoch = 20
    n_estimators = 9  # 集成模型个数
    model_list = []

    for n in range(n_estimators):
        filename = 'ensemble/' + 'ensemble' + str(epoch) + '_' + str(n) + '.h5'
        model = load_model(filename)
        model_list.append(model)

    # 评估模型
    result = ensemble_prediction(test_data)
    test_acc = sum([result[i] == test_label[i] for i in range(test_label.shape[0])]) / test_label.shape[0]
    print('accuracy:', test_acc)

    # 随机抽取五张图看与预测标签是否符合
    for i in range(5):
        n = random.randint(1,10000)
        print('predict:', fine_label_names[result[n]], ', ground truth:', fine_label_names[test_label[n]])
        plt.cla()  # 清除前面的图
        plt.imshow(test_data[n])
        title = 'predict:' + fine_label_names[result[n]] + ', ground truth:' + fine_label_names[test_label[n]]
        ax = plt.gca()  # 获取图片对象
        if result[n] == test_label[n]:
            ax.set_title(title)
        else:
            ax.set_title(title, color='red')
        plt.pause(1.0)  # 显示1秒
