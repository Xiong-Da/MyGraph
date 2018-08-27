import numpy as np

from valueNodes import *
from globalFun import *

class OpNode(ValueNode):
    def __init__(self,name,inputNodes):
        """
        :param inputNodes: it can contain np array as constant value
                        every node shape can only contain one -1 dim as sample num 
        """
        self.inputNodes=[]
        for i in range(len(inputNodes)):
            node=inputNodes[i]
            if type(node)==np.ndarray:
                node=ConstValueNode(name+"_const_"+str(i),node)
            elif isinstance(node,ValueNode)==False:
                raise Exception("input nodes should only be ValueNode or np array")
            self.inputNodes.append(node)

        self.shape=self.computeStaticOutputShape(self.inputNodes) #the first dimention of shape can be -1
        super().__init__(name,self.shape)

        self.outputBuff = None

        for node in self.inputNodes:
            node.setReferenceOp(self)

    def getInputNodes(self):
        return self.inputNodes

    def checkRunTimeOutputShape(self, runtimeShapes):
        """
        we should re alloc memory in run time if input shape is change,
        so we need a fun to compute run time out put shape,
        sub class may need overload it
        """
        _1DimValueInRuntime=[]
        for i in range(len(self.inputNodes)):
            node=self.inputNodes[i]
            staticShape=list(node.getShape())
            if -1 in staticShape:
                runtimeDimSize=runtimeShapes[i][staticShape.index(-1)]
                _1DimValueInRuntime.append(runtimeDimSize)
                staticShape[staticShape.index(-1)]=runtimeDimSize
            if list(runtimeShapes[i])!=staticShape:
                raise Exception("input data shape is diff with defined")

        if -1 not in self.shape:
            return self.shape

        if len(_1DimValueInRuntime)==0:
            raise Exception("graph define error: output shape contain -1 but input nodes")

        checkDim=_1DimValueInRuntime[0]
        for dim in _1DimValueInRuntime:
            if dim != checkDim:
                raise Exception("defualt runTime output shape check assume all -1 static dim "
                                "have same value in runtime")

        runTimeOutputShape=list(self.shape)
        runTimeOutputShape[self.shape.index(-1)]=_1DimValueInRuntime[0]  #assue they are same

        return runTimeOutputShape

    def getOutput(self):
        """
        OutputValueNode will call this fun, user should call valueNode.getValue()
        """
        inputValues = []
        inputShapes = []
        for node in self.inputNodes:
            value=node.getValue()
            inputValues.append(value)
            inputShapes.append(value.shape)

        #update buff size, as batch size may change
        runTimeOutputShape=tuple(self.checkRunTimeOutputShape(inputShapes))
        if self.outputBuff is None or self.outputBuff.shape != runTimeOutputShape:
            self.outputBuff=np.zeros(runTimeOutputShape,np.float32)

        return self.compute(inputValues,self.outputBuff)

    def compute(self,inputValues,outputBuff):
        """
        warning: numpy array obj is shared in computation,don't modi input value
        :return: output value
        """
        raise Exception("sub class should implement it")

    def computeStaticOutputShape(self, inputNodes):
        """
        compute static output shape(may contain -1)
        should also check if input is valid
        """
        raise Exception("sub class should implement it")

    def bulidGradientNodes(self,outputGradientNode,lossNodeName):
        """
        bulid gradient nodes for BP
        return [[input node,corresponding gradient node],...]
        """
        raise Exception("sub class should implement it")

    ###############################
    def __compute__(self):
        return self.getOutput()

    def getParentOp(self):
        return self

def getGradientNodeName(lossName,nodeName):
    name=lossName+"@"+nodeName
    count=0
    while True:
        if checkName(name + "_" + str(count))==False:
            count+=1
            continue
        break
    return name+"_"+str(count)

#################################################################
#implement +-*/ below

class AddOp(OpNode):
    def compute(self,inputValues,outputBuff):
        outputBuff*=0.0
        for i in range(len(inputValues)):
            outputBuff+=inputValues[i]
        return outputBuff

    def computeStaticOutputShape(self, inputNodes):
        if len(inputNodes)<=1:
            raise Exception("need at least 2 input node")
        shape=inputNodes[0].getShape()
        for node in inputNodes:
            if shape != node.getShape():
                raise Exception("invalid shape:")
        return shape

    def bulidGradientNodes(self,outputGradientNode,lossNodeName):
        """
        bulid gradient nodes for BP
        return [[input node,corresponding gradient node],...]
        """
        gradientList=[]
        for node in self.getInputNodes():
            gradientList.append([node,outputGradientNode]) #reuse gradient node
        return gradientList

class NegativeNode(OpNode):
    def compute(self,inputValues,outputBuff):
        return np.multiply(inputValues[0],-1,out=outputBuff)

    def computeStaticOutputShape(self, inputNodes):
        if len(inputNodes)!=1:
            raise Exception("need only 1 input node")
        shape=inputNodes[0].getShape()
        return shape

    def bulidGradientNodes(self,outputGradientNode,lossNodeName):
        """
        bulid gradient nodes for BP
        return [[input node,corresponding gradient node],...]
        """
        inputNode = self.getInputNodes()[0]
        gradientNode=NegativeNode(getGradientNodeName(lossNodeName,inputNode.getName()),
                                  [outputGradientNode])
        return [[inputNode,gradientNode]]

class ProductOp(OpNode):
    def compute(self,inputValues,outputBuff):
        """
        warning: numpy array obj is shared in computation,don't modi input value
        :return: output value
        """
        return np.multiply(inputValues[0],inputValues[1],out=outputBuff)

    def computeStaticOutputShape(self, inputNodes):
        """
        should also check if input is valid
        """
        if len(inputNodes)!=2:
            raise Exception("ElementWiseProduct need only 2 input node")
        if inputNodes[0].getShape()!=inputNodes[1].getShape():
            raise Exception("input nodes have diff shape")
        return inputNodes[0].getShape()

    def bulidGradientNodes(self,outputGradientNode,lossNodeName):
        """
        bulid gradient nodes for BP
        return [[input node,corresponding gradient node],...]
        """
        inputNodes=self.getInputNodes()

        gradientNode0=ProductOp(
            getGradientNodeName(lossNodeName,inputNodes[0].getName()),
            [outputGradientNode,inputNodes[1]])
        gradientNode1 = ProductOp(
            getGradientNodeName(lossNodeName, inputNodes[1].getName()),
            [outputGradientNode,inputNodes[0]])

        return [[inputNodes[0],gradientNode0],[inputNodes[1],gradientNode1]]

class ReciprocalOp(OpNode):
    """
    element wise
    """
    def compute(self,inputValues,outputBuff):
        """
        warning: numpy array obj is shared in computation,don't modi input value
        :return: output value
        """
        return np.divide(1,inputValues[0],out=outputBuff)

    def computeStaticOutputShape(self, inputNodes):
        if len(inputNodes)!=1:
            raise Exception("only need 1 input")
        return inputNodes[0].getShape()

    def bulidGradientNodes(self,outputGradientNode,lossNodeName):
        """
        bulid gradient nodes for BP
        return [[input node,corresponding gradient node],...]
        """
        inputNode=self.getInputNodes()[0]
        gradientNodeName=getGradientNodeName(lossNodeName,self.getName())

        squareNode=ProductOp(inputNode.getName() + "_square",[inputNode,inputNode])
        reciprocalSquareNode=ReciprocalOp(squareNode.getName() + "_reciprocal", [squareNode])

        nagtiveReciprocalSquareNode=NegativeNode(reciprocalSquareNode.getName() + "_nagtiveed",[reciprocalSquareNode])
        gradientNode=ProductOp(gradientNodeName,[nagtiveReciprocalSquareNode,outputGradientNode])

        return [[inputNode,gradientNode]]

class AddSmallFloatOp(OpNode):
    def compute(self,inputValues,outputBuff):
        """
        warning: numpy array obj is shared in computation,don't modi input value
        :return: output value
        """
        return inputValues[0]+1e-7

    def computeStaticOutputShape(self, inputNodes):
        """
        compute static output shape(may contain -1)
        should also check if input is valid
        """
        if len(inputNodes)!=1:
            raise Exception("only need 1 input")
        return inputNodes[0].getShape()

    def bulidGradientNodes(self,outputGradientNode,lossNodeName):
        """
        bulid gradient nodes for BP
        return [[input node,corresponding gradient node],...]
        """
        return [[self.getInputNodes()[0],outputGradientNode]]