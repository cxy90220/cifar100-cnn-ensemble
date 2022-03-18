from tensorflow.keras import layers, models
import pickle
import numpy as np
from keras.models import load_model
import time

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

# 创建CNN模型
def CNN_model(epoch):
    start = time.perf_counter()
    model = models.Sequential()
    # 底部添加作为特征提取器的卷积层

    # 卷积层1
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))

    # 卷积层2
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Activation('relu'))

    # 卷积层3
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.Activation('relu'))

    # 卷积层4
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Activation('relu'))

    # 卷积层5
    model.add(layers.Conv2D(512, (3, 3), padding='same'))
    model.add(layers.Activation('relu'))

    # 卷积层6
    model.add(layers.Conv2D(512, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))

    # 展平成一维数组
    model.add(layers.Flatten())
    # 在顶部添加密集层
    model.add(layers.Dense(800, activation='relu'))
    model.add(layers.Dense(100, activation='softmax'))

    # 显示模型的架构
    model.summary()

    # 编译和训练模型
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data, train_label, epochs=epoch, batch_size=800)
    filename = 'cnn/' + 'cnn' + str(epoch) + '.h5'
    model.save(filename)
    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))

if __name__ == '__main__':
    # 数据文件路径
    train_set_path = 'cifar-100-python/train'
    test_set_path = 'cifar-100-python/test'

    # 标签数字代表的细标签名称
    fine_label_names = [i.decode('utf8') for i in unpickle('cifar-100-python/meta')[b'fine_label_names']]

    # 读取训练集和测试集
    train_label, train_data = read_data(train_set_path)
    test_label, test_data = read_data(test_set_path)

    # 按原图片的size来reshape多维数组
    train_data = reshape_data(train_data)
    test_data = reshape_data(test_data)

    # 图像的RGB值映射到(0,1)范围
    train_data, test_data = train_data / 255.0, test_data / 255.0

    #训练模型
    epoch = 20
    CNN_model(epoch)

    # 评估模型
    filename = 'cnn/' + 'cnn' + str(epoch) + '.h5'
    model = load_model(filename)
    test_loss, test_acc = model.evaluate(test_data, test_label)
    print('accuracy:', test_acc)