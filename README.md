# cifar100-cnn-ensemble
本方法使用了六层卷积网络，针对的数据集为cifar-100，训练的批大小为800，使用的gpu为RTX-3060。
当轮数为20，集成模型数量为7时，准确率可达到59.75%，训练时长为35分钟。

单模型结果图
![image](https://user-images.githubusercontent.com/81661887/159015842-2db3efd9-d11c-46ee-bee4-051e6e07e682.png)

多模型结果图
![image](https://user-images.githubusercontent.com/81661887/159015754-d5055f31-e4fe-4b24-baf0-f5b52e526cb3.png)
