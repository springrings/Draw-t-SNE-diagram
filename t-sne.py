# coding='utf-8'

from sklearn.manifold import TSNE
from resnetv2 import *
from args import args_parser
from time import time
import matplotlib.pyplot as plt
from dataset import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
n_classes = 45
args = args_parser()

def get_data():

    net = ResNet34().cuda()
    print("loading")
    resume = torch.load(os.path.join('./checkpoints', 'ResNet34.pth'))
    net.load_state_dict(resume['model'])

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform_s1 = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        normalize,
    ])
    dst_train = RSDataset('./datasets/splits/val_split.txt', width=256,
                          height=256, transform=transform_s1)
    dataloader_train = DataLoader(dst_train, shuffle=True, batch_size=16, num_workers=12)

    cnt_total = 0

    label_list = []
    m = 0

    for j, (ims, label) in enumerate(dataloader_train):

        cnt_total += 1
        ims = ims.cuda()
        # img_tensor_s2 = img_tensor_s2.view(1, img_tensor_s2.size(0), img_tensor_s2.size(1), img_tensor_s2.size(2)).cuda()
        output = net.get_features(ims)
        output = output.view(-1,512)
        torch.save(output, "./test/" + str(m) + "myTensor.txt")
        for i in range(0,label.shape[0]):
            label_list.append(int(label[i]))
        m = m+1
    # model = torch.load(os.path.join('./checkpoints', 'ckpt.t7'))
    # print(model)
    # net = model.cuda().eval()
    for i in range(0,158):
        if i == 0:
            total = torch.load("./test/"+str(i)+"myTensor.txt").cpu()
        else:
            data = torch.load("./test/"+str(i)+"myTensor.txt").cpu()
            total = torch.cat((total,data))

    #print(type(data))
    data = total.detach().numpy()
    print(data.shape)
    #print(data.shape)
    label = label_list
    print(len(label))
    #print(label[0])
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    plt.figure(figsize=(100, 100))
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], '.',
                 color=plt.cm.Set3(label[i] ),
                 fontdict={ 'size': 30})
    plt.xticks([])
    plt.yticks([])
    # plt.title(title)
    return fig


def main():
    data, label, n_samples, n_features = get_data()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0,perplexity=50)
    t0 = time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    plt.savefig('./save_status/tsne.pdf', format='pdf')


if __name__ == '__main__':
    main()
