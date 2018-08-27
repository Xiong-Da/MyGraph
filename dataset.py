import pickle
import numpy as np
import os
from scipy.io import loadmat

######################################################################################

class TxtDataIterator():
    def __init__(self):
        with open("./datasets/linux.txt", "r") as f:
            text = f.read()

        vocab=set(text)
        countMap={c: 0 for i, c in enumerate(vocab)}
        for c in text:
            countMap[c]+=1
        dropCharList=[]
        for c in vocab:
            if countMap[c]<500:
                dropCharList.append(c)
        dropedChars=[]
        for c in text:
            if c not in dropCharList:
                dropedChars.append(c)

        self.text="".join(dropedChars)
        #self.text = text
        self.vocab = set(self.text)
        self.word_to_int_map = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_map = dict(enumerate(self.vocab))

    def getFeatureSize(self):
        return len(self.vocab)

    def toInt(self,txt):
        arr = []
        for word in txt:
            arr.append(self.word_to_int_map[word])
        return arr

    def toTxt(self,arr):
        words = []
        for index in arr:
            words.append(self.int_to_word_map[index])
        return "".join(words)

    def getNextBatch(self,n_seqs, n_steps,epochs):
        arr = np.array(self.toInt(self.text))

        featureDim=len(self.vocab)
        onehotArr = np.zeros([arr.shape[0], featureDim], np.float32)
        onehotArr[np.arange(onehotArr.shape[0]), arr] = 1.0

        batch_size = n_seqs * n_steps
        n_batches = int(len(onehotArr) / batch_size)
        if (n_batches*batch_size+n_seqs)>len(onehotArr):
            n_batches-=1
        onehotArr = onehotArr[:batch_size * n_batches+n_seqs]
        onehotArr = onehotArr.reshape((n_seqs, -1,featureDim))

        for _ in range(epochs):
            np.random.shuffle(onehotArr)
            for n in range(0, onehotArr.shape[1]//10, n_steps):
                x = onehotArr[:, n:n + n_steps,:]
                y = onehotArr[:, n+1:n + n_steps+1,:]
                yield x, y

#####################################################################################
#code modified from https://www.cnblogs.com/jimobuwu/p/9161531.html
def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']

        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        Y = np.array(Y)
        return X, Y


def getCIFAR_10(ROOT):
    """ load all of cifar """
    xs = []  # list
    ys = []

    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)

    # 测试集
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

######################################################

def getSVHNBatch(filePath):
    data=loadmat(filePath)

    image=data["X"]
    label=data["y"]

    image=np.transpose(image,(3,0,1,2))
    label=np.reshape(label,[label.shape[0]]).astype(np.int32)

    label%=10

    return image,label

def getSVHN(dataDir):
    trainFilePath = os.path.join(dataDir,"train_32x32.mat")
    testFilePath = os.path.join(dataDir, "test_32x32.mat")

    trainImage,trainLabel=getSVHNBatch(trainFilePath)
    testImage,testLabel=getSVHNBatch(testFilePath)

    return trainImage,trainLabel,testImage,testLabel

##################################################################
#code modified from https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py
import gzip
IMAGE_SIZE = 28
NUM_CHANNELS = 1

def extractMNISTImages(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        return data


def extractMNISTLabels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
    return labels

def getMNIST(dataDir):
    train_data_filename = os.path.join(dataDir,'train-images-idx3-ubyte.gz')
    train_labels_filename = os.path.join(dataDir,'train-labels-idx1-ubyte.gz')
    test_data_filename = os.path.join(dataDir,'t10k-images-idx3-ubyte.gz')
    test_labels_filename = os.path.join(dataDir,'t10k-labels-idx1-ubyte.gz')

    train_data = extractMNISTImages(train_data_filename, 60000)
    train_labels = extractMNISTLabels(train_labels_filename, 60000)
    test_data = extractMNISTImages(test_data_filename, 10000)
    test_labels = extractMNISTLabels(test_labels_filename, 10000)

    return train_data,train_labels,test_data,test_labels

#############################################################################################

def getDataset(datasetName,dataDir):
    """
    :return: trainImages,trainLabels,testImages,testLabels
    """
    dataDir=os.path.join(dataDir,datasetName)
    if datasetName=="CIFAR_10":
        return getCIFAR_10(dataDir)
    if datasetName=="SVHN":
        return getSVHN(dataDir)
    if datasetName=="MNIST":
        return getMNIST(dataDir)
    raise Exception("invalid dataset name:"+datasetName)

def getDatasetClassNum(datasetName):
    return 10


if __name__=="__main__":
    import matplotlib.pyplot as plt
    import utils
    import random
    for datasetName in ["MNIST","CIFAR_10","SVHN"]:
        train_data, train_labels, test_data, test_labels=getDataset(datasetName,"./datasets")

        train_data, train_labels=utils.shuffData(train_data, train_labels)
        test_data, test_labels=utils.shuffData(test_data, test_labels)

        for images,labels in [(train_data,train_labels),(test_data,test_labels)]:
            for _ in range(10):
                index=random.randint(0,len(images)-1)
                image=images[index]
                print(labels[index])

                if image.shape[-1]==1:
                    image=np.reshape(image,[image.shape[0],image.shape[0]])
                plt.imshow(image)