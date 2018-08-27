import numpy as np

from baseOps import *

class ReshapeOp(OpNode):
    def __init__(self, name, inputNodes):
        """
        :param name: 
        :param inputNodes: [a node obj, a np obj contain new shape]
        """
        self.runTimeInputShape=None
        self.runTimeOutputShape=None
        super().__init__(name,inputNodes)

    def checkRunTimeOutputShape(self, runtimeShapes):
        inputShape=runtimeShapes[0]
        if inputShape==self.runTimeInputShape:
            return self.runTimeOutputShape

        staticOutputShape=self.getShape()
        if -1 not in staticOutputShape:
            return staticOutputShape

        testMat=np.zeros(inputShape)
        testMat=np.reshape(testMat,staticOutputShape)

        self.runTimeInputShape=inputShape
        self.runTimeOutputShape=testMat.shape

        return self.runTimeOutputShape

    def compute(self,inputValues,outputBuff):
        """
        warning: numpy array obj is shared in computation,don't modi input value
        :return: output value
        """
        return np.reshape(inputValues[0],inputValues[1])  #shape memory of input

    def computeStaticOutputShape(self, inputNodes):
        """
        should also check if input is valid
        """
        if len(inputNodes)!=2:
            raise Exception("need 2 input node")
        oldShape = list(inputNodes[0].shape)
        newShape = inputNodes[1].getValue()  # it should be a const value node
        if -1 in oldShape:
            if -1 not in newShape:
                raise Exception("new shape should contain -1")
        else:
            try:
                a=np.zeros(oldShape)
                b=np.reshape(a,newShape)
                newShape=b.shape
            except:
                raise Exception("invalid shape")

        return tuple(newShape)

    def bulidGradientNodes(self,outputGradientNode,lossNodeName):
        """
        bulid gradient nodes for BP
        return [[input node,corresponding gradient node],...]
        """
        inputNodes = self.getInputNodes()
        gradientNode=ReshapeOp(getGradientNodeName(lossNodeName,inputNodes[0].getName()),
                               [outputGradientNode,np.array(inputNodes[0].getShape())])

        return [[inputNodes[0],gradientNode]]

class TransposOP(OpNode):
    def compute(self,inputValues,outputBuff):
        """
        warning: numpy array obj is shared in computation,don't modi input value
        :return: output value
        """
        return np.transpose(inputValues[0])  #memory resue

    def computeStaticOutputShape(self, inputNodes):
        """
        should also check if input is valid
        """
        if len(inputNodes)!=1:
            raise Exception("transpos op need only 1 input node")
        if len(inputNodes[0].getShape())!=2:
            raise Exception("transpose op need only support 2 dim array")
        return (inputNodes[0].getShape()[1],inputNodes[0].getShape()[0])

    def bulidGradientNodes(self,outputGradientNode,lossNodeName):
        """
        bulid gradient nodes for BP
        return [[input node,corresponding gradient node],...]
        """
        inputNode=self.getInputNodes()[0]
        gradientNode=TransposOP(getGradientNodeName(lossNodeName,inputNode.getName()),
                                [outputGradientNode])
        return [[inputNode,gradientNode]]

class LastDimSumOp(OpNode):
    def compute(self,inputValues,outputBuff):
        """
        warning: numpy array obj is shared in computation,don't modi input value
        :return: output value
        """
        shape=list(inputValues[0].shape)
        outputBuff=np.reshape(outputBuff,shape[:-1])

        np.sum(inputValues[0],axis=-1,out=outputBuff)

        shape[-1] = 1
        return np.reshape(outputBuff,shape)

    def computeStaticOutputShape(self, inputNodes):
        """
        compute static output shape(may contain -1)
        should also check if input is valid
        """
        if len(inputNodes)!=1:
            raise Exception("only need 1 input node")
        shape=list(inputNodes[0].getShape())
        shape[-1]=1
        return tuple(shape)

    def bulidGradientNodes(self,outputGradientNode,lossNodeName):
        """
        bulid gradient nodes for BP
        return [[input node,corresponding gradient node],...]
        """
        inputNode=self.getInputNodes()[0]
        gradientNode=self.GradientNode(getGradientNodeName(lossNodeName,inputNode.getName()),
                                       [inputNode,outputGradientNode])
        return [[inputNode,gradientNode]]

    class GradientNode(OpNode):
        def compute(self, inputValues, outputBuff):
            """
            warning: numpy array obj is shared in computation,don't modi input value
            :return: output value
            """
            shape=inputValues[0].shape
            if len(shape)==1:
                expandMat=np.ones([shape[-1]],np.float32)
                np.multiply(expandMat,inputValues[1], out=outputBuff)
            else:
                expandMat = np.ones([1, shape[-1]], np.float32)
                np.matmul(inputValues[1], expandMat, out=outputBuff)
            return outputBuff

        def computeStaticOutputShape(self, inputNodes):
            """
            compute static output shape(may contain -1)
            should also check if input is valid
            """
            if len(inputNodes) != 2:
                raise Exception("only need 2 input node")
            return inputNodes[0].getShape()

class LastDimExpandOp(OpNode):
    def compute(self,inputValues,outputBuff):
        """
        warning: numpy array obj is shared in computation,don't modi input value
        :return: output value
        """
        expandMat = np.ones([1, inputValues[1][0]], np.float32)

        return np.matmul(inputValues[0], expandMat, out=outputBuff)

    def computeStaticOutputShape(self, inputNodes):
        """
        compute static output shape(may contain -1)
        should also check if input is valid
        """
        if len(inputNodes)!=2:
            raise Exception("only need 2 input nodes")
        if len(inputNodes[1].getShape())!=1:
            raise Exception("last input node should represent expend size")

        shape=list(inputNodes[0].getShape())
        if shape[-1]!=1:
            raise Exception("last dim of frist input node should be 1")
        if type(inputNodes[1])!=ConstValueNode:
            shape[-1]=-1
        shape[-1]=inputNodes[1].getValue()[0]
        return shape

    def bulidGradientNodes(self,outputGradientNode,lossNodeName):
        """
        bulid gradient nodes for BP
        return [[input node,corresponding gradient node],...]
        """
        inputNode=self.getInputNodes()[0]
        gradientNode=LastDimSumOp(getGradientNodeName(lossNodeName,inputNode.getName()),
                                  [outputGradientNode])
        return [[inputNode,gradientNode]]

class ConcatOp(OpNode):
    def compute(self,inputValues,outputBuff):
        """
        warning: numpy array obj is shared in computation,don't modi input value
        :return: output value
        """
        curDim=0
        for value in inputValues:
            outputBuff[:,curDim:curDim+value.shape[-1]]=value
            curDim+=value.shape[-1]
        return outputBuff

    def computeStaticOutputShape(self, inputNodes):
        """
        compute static output shape(may contain -1)
        should also check if input is valid
        """
        shape=list(inputNodes[0].getShape(0))
        if len(shape)!=2:
            raise Exception("only support 2 dim mats")
        shape[-1]=0
        for node in inputNodes:
            shape[-1]+=node.getShape([-1])
        return shape

    def bulidGradientNodes(self,outputGradientNode,lossNodeName):
        """
        bulid gradient nodes for BP
        return [[input node,corresponding gradient node],...]
        """
        inputNodes=self.getInputNodes()

        curDim = 0
        retList=[]
        for inputNode in inputNodes:
            shape=inputNode.getShape()
            gradientNode=self.GradientNode(getGradientNodeName(lossNodeName,inputNode.getName()),
                                           [outputGradientNode,np.array([curDim]),np.array([curDim+shape[-1]])])
            curDim+=shape[-1]
            retList.append([inputNode,gradientNode])
        return retList

    class GradientNode(OpNode):
        def compute(self, inputValues, outputBuff):
            """
            warning: numpy array obj is shared in computation,don't modi input value
            :return: output value
            """
            outputBuff=inputValues[0][:,inputValues[1][0]:inputValues[2][0]]
            return outputBuff

        def computeStaticOutputShape(self, inputNodes):
            shape=list(inputNodes[0].getShape())
            shape[-1]=inputNodes[2].getValue()[0]-inputNodes[1].getValue()[0]
            return shape