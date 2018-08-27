import numpy as np

from baseOps import *
from shapeOps import *
from mathOps import *
from globalFun import *

from valueNodes import *

def softmax(inputNode,name):
    if len(inputNode.getShape())!=2:
        raise Exception("softmax only need 1 input node")

    exponentNode=ExponentOp(inputNode.getName() + "_exponent", [inputNode])
    sumExponentNode=LastDimSumOp(exponentNode.getName()+"_sum",[exponentNode])
    expandSumNode=LastDimExpandOp(sumExponentNode.getName()+"_expand",[sumExponentNode,
                                                  np.array([exponentNode.getShape()[-1]])])

    addSmallSum=AddSmallFloatOp(expandSumNode.getName() + "_addSmall", [expandSumNode])
    reciprocalSumNode=ReciprocalOp(addSmallSum.getName()+"_reciprocal",[addSmallSum])

    return ProductOp(name,[exponentNode,reciprocalSumNode])

def meanCrossEntropy(predictProb, onehotLabel, batchSizePlaceholder,name):
    if len(predictProb.getShape())!=2:
        raise Exception("only support 2 dim mat")

    predictProb=AddSmallFloatOp(predictProb.getName()+"_addSmall",[predictProb])
    logNode=LogNode(predictProb.getName()+"_log",[predictProb])

    productNode=ProductOp(name+"_product",[logNode,onehotLabel])
    negProductNode=NegativeNode(productNode.getName()+"_negtive",[productNode])

    sumNode1=LastDimSumOp(negProductNode.getName()+"_sum",[negProductNode])
    reshapedSum1=ReshapeOp(sumNode1.getName()+"_shape",[sumNode1,np.array(sumNode1.getShape()[:-1])])
    sumNode2=LastDimSumOp(reshapedSum1.getName()+"_sum",[reshapedSum1])

    reciprocalBatchSize=getNodeByName(batchSizePlaceholder.getName()+"_reciprocal")
    if reciprocalBatchSize==None:
        reciprocalBatchSize=ReciprocalOp(batchSizePlaceholder.getName()+"_reciprocal",
                                         [batchSizePlaceholder])

    return ProductOp(name,[sumNode2,reciprocalBatchSize])

def denseLayer(inputNode,outputChannel,name):
    inputShape=inputNode.getShape()
    if len(inputShape)!=2:
        raise Exception("invalid input shape")

    weightInitValue=np.random.randn(inputShape[1], outputChannel) / np.sqrt(inputShape[1])
    weight=WeightValueNode(name+"_weight",np.array(weightInitValue,np.float32))

    # weight = WeightValueNode(name + "_weight",np.ones([inputShape[1], outputChannel],np.float32))

    bias=WeightValueNode(name+"_bias",np.zeros([outputChannel],np.float32))

    out=MatMuilNode(name+"_mul",[inputNode,weight])
    out=BiasAddNode(name,[out,bias])

    return out
