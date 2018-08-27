import numpy as np

from baseOps import *
from shapeOps import *
from mathOps import *
from globalFun import *
from optimizer import *

from nn import *
from CNN import *
from LSTM import *

import utils
import dataset

BATCH_SIZE=128

def getOneHotLabel(label,classNum):
    oneHotLabel=np.zeros([label.shape[0],classNum],np.float32)
    oneHotLabel[np.arange(label.shape[0]),label]=1.0
    return oneHotLabel

def testFullConnectedNNOnMnist():
    trainData, trainLabel, testData, testLabel=dataset.getDataset("MNIST","./datasets")

    trainData=trainData.astype(np.float32)
    testData=testData.astype(np.float32)
    utils.normalizeData(trainData)
    utils.normalizeData(testData)

    trainDataIterator=utils.DatasetLiterator(trainData,trainLabel,50,BATCH_SIZE)
    testDataIterator=utils.DatasetLiterator(testData,testLabel,1,BATCH_SIZE)

    imagePlaceholder=Placeholder("image",[-1,28*28])
    labelPlaceholder=Placeholder("label",[-1,10])
    bathSizePlaceholder=Placeholder("batch size",[1])
    learningRatePlaceholder=Placeholder("learning rate",[1])

    output=imagePlaceholder

    output=denseLayer(output,128,"layer1")
    output=ReluNode("relu1",[output])

    output=denseLayer(output,128,"layer2")
    output = ReluNode("relu2", [output])

    logits=denseLayer(output,10,"layer3")
    predictLabel=softmax(logits,"softmax")

    loss=meanCrossEntropy(predictLabel,labelPlaceholder,bathSizePlaceholder,"loss")

    gradientPairs=minimize(loss)
    trainOp=SGDNode("train op",gradientPairs,
                            learningRatePlaceholder)

    step=0
    learningRate=np.array([1e-2])
    bathSize=np.array([BATCH_SIZE])
    for image,label in trainDataIterator.getNextBatch():
        image=np.reshape(image,[-1,28*28])
        label=getOneHotLabel(label,10)

        feedDic={imagePlaceholder.getName():image,
                 labelPlaceholder.getName():label,
                 bathSizePlaceholder.getName():bathSize,
                 learningRatePlaceholder.getName():learningRate}

        setFeedDic(feedDic)
        trainOp.getValue()

        step+=1

        if step%100==0:
            print("loss:"+str(loss.getValue()[0]))

    testCount=0
    errorCount=0
    for image,label in testDataIterator.getNextBatch():
        image=np.reshape(image,[-1,28*28])

        feedDic = {imagePlaceholder.getName(): image}

        setFeedDic(feedDic)
        predict=predictLabel.getValue()
        predict=np.argmax(predict,-1)

        testCount+=len(label)
        errorCount+=sum((predict!=label).astype(np.int32))

    print("\n\n")
    print("error rate:"+str(errorCount/testCount))

def testCNNOnCIAFR():
    trainData, trainLabel, testData, testLabel = dataset.getDataset("SVHN", "./datasets")

    trainData = trainData.astype(np.float32)
    testData = testData.astype(np.float32)
    utils.normalizeData(trainData)
    utils.normalizeData(testData)

    trainDataIterator = utils.DatasetLiterator(trainData, trainLabel, 10, BATCH_SIZE)
    testDataIterator = utils.DatasetLiterator(testData, testLabel, 1, BATCH_SIZE)

    imagePlaceholder = Placeholder("image", [-1,32,32,3])
    labelPlaceholder = Placeholder("label", [-1, 10])
    bathSizePlaceholder = Placeholder("batch size", [1])
    learningRatePlaceholder = Placeholder("learning rate", [1])

    output = imagePlaceholder

    output=convLayer(output,3,2,32,"layer1")
    output=ReluNode("relu1",[output])

    output = convLayer(output, 3, 2, 64, "layer2")
    output = ReluNode("relu2", [output])

    output = convLayer(output, 3, 2, 128, "layer3")
    output = ReluNode("relu3", [output])

    print("last conv feature size:"+str(output.getShape()))
    avgPoolSize=max(output.getShape()[1],output.getShape()[2])
    output = avgPool(output,avgPoolSize,1,"globalPool")
    output = ReshapeOp("reshape",[output,np.array([-1,128])])

    logits = denseLayer(output, 10, "logits")
    predictLabel = softmax(logits, "softmax")

    loss = meanCrossEntropy(predictLabel, labelPlaceholder, bathSizePlaceholder, "loss")

    gradientPairs = minimize(loss)
    trainOp = MomentumSGDNode("train op", gradientPairs,
                      learningRatePlaceholder,ConstValueNode("moment",np.array([0.9])))

    step = 0
    learningRate = np.array([1e-2])
    bathSize = np.array([BATCH_SIZE])
    nodes=[node for node in getNodeByConstructSeq()
           if "const" not in node.getName() and "batch" not in node.getName()]
    for image, label in trainDataIterator.getNextBatch():
        label = getOneHotLabel(label, 10)

        feedDic = {imagePlaceholder.getName(): image,
                   labelPlaceholder.getName(): label,
                   bathSizePlaceholder.getName(): bathSize,
                   learningRatePlaceholder.getName(): learningRate}

        setFeedDic(feedDic)
        trainOp.getValue()

        # for node in nodes:
        #     name=node.getName()
        #     value=node.getValue()
        #     shape=value.shape
        #     pass

        step += 1
        print("step:"+str(step)+"   loss:" + str(loss.getValue()[0]))

    testCount = 0
    errorCount = 0
    for image, label in testDataIterator.getNextBatch():
        feedDic = {imagePlaceholder.getName(): image}
        setFeedDic(feedDic)
        predict = predictLabel.getValue()
        predict = np.argmax(predict, -1)

        testCount += len(label)
        errorCount += sum((predict != label).astype(np.int32))

    print("\n\n")
    print("error rate:" + str(errorCount / testCount))

def testLSTM():
    dataIterator=dataset.TxtDataIterator()
    featureDim=dataIterator.getFeatureSize()

    batchSize=256
    timeStep=32
    hidenDim=128

    rnnParam=getGRUParamNodes("gateParam",featureDim,hidenDim)
    denseParam=getDenselayerParamNodes("dense",hidenDim,featureDim)

    learningRatePlaceholder = Placeholder("learning rate", [1])
    bathSizePlaceholder = Placeholder("batch size", [1])
    initialStatePlaceholder=Placeholder("init",[-1,hidenDim])
    inputPlaceholders=[]
    hidenStates=[]
    labelPlaceholders=[]
    outputs=[]

    lastHidenState=initialStatePlaceholder
    for i in range(timeStep):
        labelPlaceholder=Placeholder("label"+str(i),[-1,featureDim])
        labelPlaceholders.append(labelPlaceholder)

        inputPlaceholder=Placeholder("input"+str(i),[-1,featureDim])
        inputPlaceholders.append(inputPlaceholder)

        lastHidenState=oneStep("step"+str(i),lastHidenState,inputPlaceholder,rnnParam)
        hidenStates.append(lastHidenState)

        logits=denseWithGivenParam("dense"+str(i),lastHidenState,denseParam)
        predict=softmax(logits,"softmax"+str(i))
        outputs.append(predict)

    losses=[]
    for i in range(len(outputs)):
        output=outputs[i]
        label=labelPlaceholders[i]
        loss=meanCrossEntropy(output, label, bathSizePlaceholder, "loss"+str(i))
        losses.append(loss)

    loss=AddOp("finalLoss",losses)
    gradientPairs = minimize(loss)
    trainOp = MomentumSGDNode("train op", gradientPairs,
                              learningRatePlaceholder, ConstValueNode("moment", np.array([0.9],dtype=np.float32)))

    nodes=getNodeByConstructSeq()
    debugNode=[node for node in getNodeByConstructSeq()  \
               if "const" not in node.getName() \
               and "batch" not in node.getName() \
               and "place" not in node.getName() \
               and "Param" not in node.getName()]
    step=0
    for _ in range(1):
        lastState=np.zeros([batchSize,hidenDim],np.float32)
        for x,y in dataIterator.getNextBatch(batchSize,timeStep,1):
            feedDic={learningRatePlaceholder.getName():np.array([1e-3],np.float32),
                     bathSizePlaceholder.getName():np.array([batchSize]),
                     initialStatePlaceholder.getName():lastState}

            for i in range(len(inputPlaceholders)):
                inputPlaceholder=inputPlaceholders[i]
                labelPlaceholder=labelPlaceholders[i]
                feedDic[inputPlaceholder.getName()]=x[:,i,:]
                feedDic[labelPlaceholder.getName()]=y[:,i,:]

            setFeedDic(feedDic)
            for node in debugNode:
                name=node.getName()
                value=node.getValue()
            trainOp.getValue()
            lastState=hidenStates[-1].getValue()

            print(str(step)+":"+str(loss.getValue()))
            step+=1

    ###############################################################
    #try generate sample
    startCode="""
    while true{
        if(areYouOk()==false): //are you a gay
            continue;
        else:{
            printf("yes we can!");
            break;
        }
    }
    """
    lastChar=None
    sampleList=dataIterator.toInt(startCode)
    lastState = np.zeros([1, hidenDim],np.float32)
    for i in range(500):
        inputChar=sampleList[i]
        inputFaeture=np.zeros([1,featureDim],dtype=np.float32)
        inputFaeture[0][inputChar]=1.0

        feedDic = {}
        feedDic[inputPlaceholders[0].getName()] = inputFaeture
        feedDic[initialStatePlaceholder.getName()] = lastState
        setFeedDic(feedDic)

        lastState = hidenStates[0].getValue()
        output=outputs[0].getValue()[0]
        output=np.argmax(output)

        if i == len(sampleList)-1:
            sampleList.append(output)

    txt=dataIterator.toTxt(sampleList)
    print("\n"*2+"#"*20)
    print(txt)

    print("\n"*2)
    print(sampleList)

if __name__=="__main__":
    #testFullConnectedNNOnMnist()
    #testCNNOnCIAFR()
    testLSTM()