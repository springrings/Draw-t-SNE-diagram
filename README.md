# Draw-t-SNE-diagram
Visualization of classification results through t-sne

使用t-sne把训练好的模型的分类的结果进行可视化，这里是将resnet34的最后一个block的输出进行可视化，因为环境问题，直接进行可视化会导致我的空间溢出，所以在这个项目先将特征张量存储到本地txt文件，再读取，实际上只需要保证t-sne.py中get_data函数里的data是各个图片的特征张量，label是对应的各个图片的标签就可以了；图片的获取方式为从txt读取，可自行更改。

需要更改的参数为：预训练模型及路径，数据集路径，（如果需要的话）特征存储txt路径

Use t-sne to visualize the classification results of the trained model. Here is to visualize the output of the last block of resnet34. Because of environmental problems, direct visualization will lead to my space overflow. Therefore, in this project, first store the feature tensor in the local TXT file and then read it. In fact, just guarantee the data in the get_data() function is the characteristic tensor of each picture, and the label is the label of each picture; The image acquisition method is to read from TXT, which can be changed by yourself.

The parameters to be changed are: pretrained model and path, data set path, and (if necessary) feature storage TXT path

![image](https://user-images.githubusercontent.com/70803368/146488373-1af89876-014e-49dc-b583-28184151c5e8.png)
