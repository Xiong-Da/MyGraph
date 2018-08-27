import numpy as np

from baseOps import *
from shapeOps import *
from mathOps import *
from globalFun import *

from nn import *

def getWeightNode(name,inputDim,outputDim):
    weightInitValue = np.random.randn(inputDim, outputDim) / np.sqrt(inputDim)
    weight = WeightValueNode(name, np.array(weightInitValue, np.float32))
    return weight

def getBiasNode(name,outputDim):
    bias = WeightValueNode(name, np.zeros([outputDim], np.float32))
    return bias

def getDenselayerParamNodes(name,inputDim,outputDim):
    return [getWeightNode(name+"_weight",inputDim,outputDim),
            getBiasNode(name+"_bias",outputDim)]

def getGateParam(name,hidenDim,inputDim,outputDim):
    hidenParam=getWeightNode(name+"_hiden",hidenDim,outputDim)
    inputParam=getWeightNode(name+"_input",inputDim,outputDim)
    bias=getBiasNode(name+"_bias",outputDim)

    return [hidenParam,inputParam,bias]

def getGRUParamNodes(name,inputDim,outputDim):
    paramOfGateZ=getGateParam(name+"_gateZ",outputDim,inputDim,1)
    paramOfGateR=getGateParam(name+"_gateR",outputDim,inputDim,1)
    paramOfHidenState=getGateParam(name+"_hiden",outputDim,inputDim,outputDim)

    return [paramOfGateZ, paramOfGateR, paramOfHidenState]

def computGate(name,hidenState, input,param):
    """
    :param hidenState: 2 dim mat [batch,featureDim]
    :param input: 2 dim mat [batch,featureDim]
    :param param: return of getGateParam fun
    :return: 2 dim mat [batch,featureDim]
    """
    hidenParam = param[0]
    inputParam = param[1]
    bias = param[2]

    hiden=MatMuilNode(name+"_hiden",[hidenState,hidenParam])
    input=MatMuilNode(name+"_input",[input,inputParam])
    plus=AddOp(name+"add",[hiden,input])

    return BiasAddNode(name,[plus,bias])

class OneSubOp(OpNode):
    def compute(self,inputValues,outputBuff):
        """
        warning: numpy array obj is shared in computation,don't modi input value
        :return: output value
        """
        np.subtract(1,inputValues[0],out=outputBuff)
        return outputBuff

    def computeStaticOutputShape(self, inputNodes):
        """
        compute static output shape(may contain -1)
        should also check if input is valid
        """
        return inputNodes[0].getShape()

    def bulidGradientNodes(self,outputGradientNode,lossNodeName):
        input=self.getInputNodes()[0]
        gradient=NegativeNode(getGradientNodeName(lossNodeName,input.getName()),[outputGradientNode])
        return [[input,gradient]]

def oneStep(name,hidenState, input,params):
    paramOfGateZ = params[0]
    paramOfGateR = params[1]
    paramOfHidenState = params[2]

    gateZ=computGate(name+"_gateZ",hidenState,input,paramOfGateZ)
    gateZ=SigmoidNode(gateZ.getName() +"Sigmoid",[gateZ])
    gateZ=LastDimExpandOp(gateZ.getName()+"Expand",[gateZ,np.array([hidenState.getShape()[-1]])])
    oneSubGateZ=OneSubOp(gateZ.getName()+"_1Sub",[gateZ])

    gateR = computGate(name + "_gateR", hidenState, input, paramOfGateR)
    gateR = SigmoidNode(gateR.getName() + "_Sigmoid", [gateR])
    gateR = LastDimExpandOp(gateR.getName()+"_Expand", [gateR, np.array([hidenState.getShape()[-1]])])

    productHidenState=ProductOp(name+"_productHidenState",[hidenState,gateR])

    newState=computGate(name+"_newState",productHidenState,input,paramOfHidenState)
    newState=TanhNode(name+"_tanh",[newState])
    newState=ProductOp(name+"_newSate",[newState,gateZ])
    oldState=ProductOp(name+"_oldState",[hidenState,oneSubGateZ])
    newState=AddOp(name,[newState,oldState])

    return newState

def denseWithGivenParam(name,input,params):
    """
    :param input: 
    :param params: return of getDenselayerParamNodes
    """
    weight=params[0]
    bias=params[1]

    out = MatMuilNode(name + "_mul", [input, weight])
    out = BiasAddNode(name, [out, bias])

    return out
