import random
import numpy as np

###########################################################

def shuffData(images,labels):
    index=[i for i in range(len(labels))]
    random.shuffle(index)

    return images[index],labels[index]

def normalizeData(images):
    means = []
    stds = []

    # for every channel in image(assume this is last dimension)
    for ch in range(images.shape[-1]):
        means.append(np.mean(images[:, :, :, ch]))
        stds.append(np.std(images[:, :, :, ch]))

    for i in range(images.shape[-1]):
        images[:, :, :, i] = ((images[:, :, :, i] - means[i]) / stds[i])

def _augmentImage(image, pad,needflip):
    """Perform zero padding, randomly crop image to original size,
    maybe mirror horizontally"""
    init_shape = image.shape
    new_shape = [init_shape[0] + pad * 2,
                 init_shape[1] + pad * 2,
                 init_shape[2]]
    zeros_padded = np.zeros(new_shape)
    zeros_padded[pad:init_shape[0] + pad, pad:init_shape[1] + pad, :] = image
    # randomly crop to original size
    init_x = np.random.randint(0, pad * 2)
    init_y = np.random.randint(0, pad * 2)
    cropped = zeros_padded[
        init_x: init_x + init_shape[0],
        init_y: init_y + init_shape[1],
        :]
    flip = random.getrandbits(1) and needflip
    if flip:
        cropped = cropped[:, ::-1, :]
    return cropped

def augmentData(initial_images,needflip):
    new_images = np.zeros(initial_images.shape)
    for i in range(initial_images.shape[0]):
        new_images[i] = _augmentImage(initial_images[i], pad=4,needflip=needflip)
    return new_images

#above code come from https://github.com/ikhlestov/vision_networks/tree/master/data_providers

###############################################

class DatasetLiterator:
    def __init__(self,images,labels,epochs,batchSize):
        self.images=images
        self.labels=np.array(labels,np.int32)
        self.epochs=epochs
        self.batchSize=batchSize

        if batchSize>len(images):
            raise Exception("too big batch size")

    def getNextBatch(self):
        index=[]

        for i in range(self.epochs):
            index+=[i for i in range(len(self.images))]
            random.shuffle(index)

            while len(index)>=self.batchSize:
                batchIndex=index[:self.batchSize]
                index=index[self.batchSize:]

                batchImage=self.images[batchIndex]
                batchLabel=self.labels[batchIndex]

                yield batchImage,batchLabel

        #yield last images, batch size is changed
        if len(index)!=0:
            yield self.images[index],self.labels[index]


class AugmentDatasetLiterator:
    def __init__(self,images,labels,epochs,batchSize,needflip=True):
        self.iterator=DatasetLiterator(images,labels,epochs,batchSize)
        self.needFlip=needflip

    def getNextBatch(self):
        for batchImage,batchLabel in self.iterator.getNextBatch():
            yield augmentData(batchImage,self.needFlip), batchLabel

#################################################################################

class LocalLearningRateTuner():
    """
    base on validate test err in slide window
    """
    def __init__(self,startLearningRate=1e-3,decayFactor=0.5,threshold=0.02,slideWindow=4,maxCoolDown=3,maxDontSave=6,minLearningRate=5e-5):
        self.learningRateRecoder=[]

        self.curLearningRate=startLearningRate
        self.decayFactor=decayFactor
        self.threshold=threshold
        self.slidWindow=slideWindow
        self.validateErrs=[]

        self.maxCoolDown=maxCoolDown
        self.maxDontSave=maxDontSave
        self.minLearningRate=minLearningRate
        self.curLowestErr=1
        self.dontSaveCount=0

        self.coolDown=0

    def updateValidateErr(self, err):
        if self.curLowestErr>err:
            self.curLowestErr=err
            self.dontSaveCount=0
        else:
            self.dontSaveCount+=1

        self.validateErrs.append(err)
        if len(self.validateErrs)<self.slidWindow:
            return
        if len(self.validateErrs)>self.slidWindow:
            self.validateErrs.pop(0)

        oldAvgErr=np.mean(self.validateErrs[:int(len(self.validateErrs) / 2)])
        newAvgErr =np.mean(self.validateErrs[int(len(self.validateErrs) / 2):])
        relativeDec=(oldAvgErr-newAvgErr) / oldAvgErr

        # print("\n" * 2 + "*" * 20)
        # print(self.validateErrs)
        # print("oldAvgErr:"+str(oldAvgErr)+"  newAvgErr:"+str(newAvgErr))
        # print("relativeDec:"+str(relativeDec))

        self.coolDown -= 1
        if relativeDec<self.threshold:
            if self.coolDown>0:
                #print("keep learning rate to:" + str(self.curLearningRate))
                pass
            else:
                self.coolDown=self.maxCoolDown
                self.curLearningRate*=self.decayFactor
                if self.curLearningRate<self.minLearningRate:
                    self.curLearningRate=self.minLearningRate
                #print("decay learning rate to:"+str(self.curLearningRate))
        else:
            #print("keep learning rate to:"+str(self.curLearningRate))
            pass

    def getLearningRate(self):
        self.learningRateRecoder.append(self.curLearningRate)
        return self.curLearningRate

    def isShouldSave(self):
        return self.dontSaveCount==0

    def isShouldStop(self):
        if self.dontSaveCount>=self.maxDontSave:
            return True
        return False

    def getFixTuner(self):
        return FixLearningRateTuner(self.learningRateRecoder[:-(self.dontSaveCount-1)],isEarlyStop=False)

class GlobalLearningRateTuner():
    """
    base on global validate test err
    """
    def __init__(self,startLearningRate=1e-3,decayFactor=0.5,threshold=3,maxDontSave=7,minLearningRate=5e-5):
        self.learningRateRecoder=[]

        self.curLearningRate = startLearningRate
        self.decayFactor = decayFactor
        self.threshold = threshold

        self.maxDontSave = maxDontSave
        self.minLearningRate = minLearningRate
        self.curLowestErr = 1
        self.dontSaveCount = 0

    def updateValidateErr(self, err):
        if self.curLowestErr>err:
            self.curLowestErr=err
            self.dontSaveCount=0
            #print("err:"+str(err)+" cur lowest")
            return
        else:
            self.dontSaveCount+=1

        if self.dontSaveCount % self.threshold==0 and self.dontSaveCount!=0:
            self.curLearningRate *= self.decayFactor
            if self.curLearningRate < self.minLearningRate:
                self.curLearningRate = self.minLearningRate
            #print("err:"+str(err)+" decay learning rate to:"+str(self.curLearningRate))

    def getLearningRate(self):
        self.learningRateRecoder.append(self.curLearningRate)
        return self.curLearningRate

    def isShouldSave(self):
        return self.dontSaveCount==0

    def isShouldStop(self):
        if self.dontSaveCount>=self.maxDontSave:
            return True
        return False

    def getFixTuner(self):
        return FixLearningRateTuner(self.learningRateRecoder[:-(self.dontSaveCount-1)],isEarlyStop=False)

class FixLearningRateTuner():
    def __init__(self,learningRateIndex,isEarlyStop=False):
        self.checkCount=0
        self.learningRateIndex=learningRateIndex

        self.isEarlyStop=isEarlyStop
        self.curLowestErr=1.0
        self.dontSaveCount=0

    def updateValidateErr(self, err):
        self.checkCount+=1
        if err<self.curLowestErr:
            self.curLowestErr=err
            self.dontSaveCount=0
        else:
            self.dontSaveCount+=1

    def getLearningRate(self):
        if self.checkCount<len(self.learningRateIndex):
            return self.learningRateIndex[self.checkCount]
        return self.learningRateIndex[-1]

    def isShouldSave(self):
        if self.isEarlyStop==True and self.dontSaveCount!=0:
            return False
        return True

    def isShouldStop(self):
        return len(self.learningRateIndex) == self.checkCount


def getFixLearningRateTuner(checkCountList, learningRateList, isEarlyStop):
    learningRateIndex = []
    for i in range(len(checkCountList)):
        for _ in range(checkCountList[i]):
            learningRateIndex.append(learningRateList[i])
    return FixLearningRateTuner(learningRateIndex,isEarlyStop)