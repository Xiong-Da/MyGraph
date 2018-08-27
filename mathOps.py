import numpy as np

from baseOps import *
from shapeOps import *

class MatMuilNode(OpNode):
    def compute(self,inputValues,outputBuff):
        """
        warning: numpy array obj is shared in computation,don't modi input value
        :return: output value
        """
        return np.matmul(inputValues[0],inputValues[1],out=outputBuff)

    def computeStaticOutputShape(self, inputNodes):
        """
        compute static output shape(may contain -1)
        should also check if input is valid
        """
        if len(inputNodes)!=2:
            raise Exception("need 2 input node")
        for node in inputNodes:
            if len(node.getShape())!=2:
                raise Exception("nodes should be 2 dim mat")
        if inputNodes[0].getShape()[1]!=inputNodes[1].getShape()[0]:
            raise Exception("invalid shape")

        shape=list(inputNodes[0].getShape())
        shape[-1]=inputNodes[1].getShape()[1]

        return tuple(shape)

    def bulidGradientNodes(self,outputGradientNode,lossNodeName):
        """
        C=AB
        
        bulid gradient nodes for BP
        return [[input node,corresponding gradient node],...]
        """
        inputNodes=self.getInputNodes()
        A=inputNodes[0]
        B=inputNodes[1]

        transposedB=getNodeByName(B.getName()+"_transposed")  #B is weight mat, may share in diff op
        if transposedB==None:
            transposedB=TransposOP(B.getName()+"_transposed",[B])
        gradientNodeA=MatMuilNode(getGradientNodeName(lossNodeName,A.getName()),
                                  [outputGradientNode,transposedB])

        transposedA = getNodeByName(A.getName() + "_transposed") #node may share in diff op
        if transposedA==None:
            transposedA = TransposOP(A.getName() + "_transposed",[A])
        gradientNodeB = MatMuilNode(getGradientNodeName(lossNodeName, B.getName()),
                                    [transposedA,outputGradientNode])

        return [[A,gradientNodeA],[B,gradientNodeB]]

class BiasAddNode(OpNode):
    def compute(self,inputValues,outputBuff):
        """
        warning: numpy array obj is shared in computation,don't modi input value
        :return: output value
        """
        return np.add(inputValues[0],inputValues[1],out=outputBuff)

    def computeStaticOutputShape(self, inputNodes):
        """
        compute static output shape(may contain -1)
        should also check if input is valid
        """
        if len(inputNodes)!=2:
            raise Exception("need 2 input node")
        if len(inputNodes[0].getShape())!=2 or len(inputNodes[1].getShape())!=1:
            raise Exception("input node shape error")
        return inputNodes[0].getShape()

    def bulidGradientNodes(self,outputGradientNode,lossNodeName):
        """
        bulid gradient nodes for BP
        return [[input node,corresponding gradient node],...]
        """
        inpuNodes=self.getInputNodes()

        transposedBPGradient=TransposOP(outputGradientNode.getName()+"_transpose",
                                        [outputGradientNode])
        sumTransBPGrad=LastDimSumOp(transposedBPGradient.getName()+"_sum",
                                    [transposedBPGradient])
        grandient=ReshapeOp(sumTransBPGrad.getName()+"_reshape",
                            [sumTransBPGrad,np.array([-1])])

        return [[inpuNodes[0],outputGradientNode],[inpuNodes[1],grandient]]


class ReluNode(OpNode):
    def __init__(self, name, inputNodes):
        super().__init__(name, inputNodes)
        self.mask=None

    def compute(self,inputValues,outputBuff):
        """
        warning: numpy array obj is shared in computation,don't modi input value
        :return: output value
        """
        isBigThan0=inputValues[0]>0
        self.mask=isBigThan0.astype(np.float32)
        return inputValues[0]*self.mask

    def computeStaticOutputShape(self, inputNodes):
        """
        compute static output shape(may contain -1)
        should also check if input is valid
        """
        if len(inputNodes)!=1:
            raise Exception("only need 1 input node")
        return inputNodes[0].getShape()

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
            isBigThan0 = inputValues[0] > 0
            mask = isBigThan0.astype(np.float32)
            return inputValues[1] * mask

        def computeStaticOutputShape(self, inputNodes):
            """
            compute static output shape(may contain -1)
            should also check if input is valid
            """
            if len(inputNodes) != 2:
                raise Exception("only need 2 input node")
            return inputNodes[0].getShape()

class ExponentOp(OpNode):
    def compute(self,inputValues,outputBuff):
        """
        warning: numpy array obj is shared in computation,don't modi input value
        :return: output value
        """
        return np.exp(inputValues[0],out=outputBuff)

    def computeStaticOutputShape(self, inputNodes):
        """
        compute static output shape(may contain -1)
        should also check if input is valid
        """
        if len(inputNodes)!=1:
            raise Exception("only need 1 input node")
        return inputNodes[0].getShape()

    def bulidGradientNodes(self,outputGradientNode,lossNodeName):
        """
        bulid gradient nodes for BP
        return [[input node,corresponding gradient node],...]
        """
        inputNode=self.getInputNodes()[0]
        gradientNode=ProductOp(getGradientNodeName(lossNodeName,inputNode.getName()),
                               [self,outputGradientNode])
        return [[inputNode,gradientNode]]

class LogNode(OpNode):
    def compute(self,inputValues,outputBuff):
        """
        warning: numpy array obj is shared in computation,don't modi input value
        :return: output value
        """
        return np.log(inputValues[0],out=outputBuff)

    def computeStaticOutputShape(self, inputNodes):
        """
        compute static output shape(may contain -1)
        should also check if input is valid
        """
        if len(inputNodes)!=1:
            raise Exception("only need 1 node")
        return inputNodes[0].getShape()

    def bulidGradientNodes(self,outputGradientNode,lossNodeName):
        """
        bulid gradient nodes for BP
        return [[input node,corresponding gradient node],...]
        """
        inputNode=self.getInputNodes()[0]
        gradientNodeName = getGradientNodeName(lossNodeName, self.getName())

        reciprocalNode=ReciprocalOp(inputNode.getName()+"_recip",[inputNode])
        gradientNode=ProductOp(gradientNodeName,[outputGradientNode,reciprocalNode])

        return [[inputNode,gradientNode]]

class SigmoidNode(OpNode):
    def compute(self,inputValues,outputBuff):
        """
        warning: numpy array obj is shared in computation,don't modi input value
        :return: output value
        """
        np.exp(inputValues[0],out=outputBuff)
        outputBuff+=1
        np.divide(1.0,outputBuff,out=outputBuff)
        return outputBuff

    def computeStaticOutputShape(self, inputNodes):
        """
        compute static output shape(may contain -1)
        should also check if input is valid
        """
        return inputNodes[0].getShape()

    def bulidGradientNodes(self,outputGradientNode,lossNodeName):
        """
        bulid gradient nodes for BP
        return [[input node,corresponding gradient node],...]
        """
        inputNode=self.getInputNodes()[0]
        gradientNode=self.GradientNode(getGradientNodeName(lossNodeName,inputNode.getName()),
                                       [self,outputGradientNode])
        return [[inputNode,gradientNode]]

    class GradientNode(OpNode):
        def compute(self, inputValues, outputBuff):
            """
            warning: numpy array obj is shared in computation,don't modi input value
            :return: output value
            """
            np.subtract(1,inputValues[0],out=outputBuff)
            outputBuff*=inputValues[0]
            outputBuff*=inputValues[1]
            return outputBuff

        def computeStaticOutputShape(self, inputNodes):
            """
            compute static output shape(may contain -1)
            should also check if input is valid
            """
            return inputNodes[0].getShape()

class TanhNode(OpNode):
    def compute(self,inputValues,outputBuff):
        """
        warning: numpy array obj is shared in computation,don't modi input value
        :return: output value
        """
        np.tanh(inputValues[0],out=outputBuff)
        return outputBuff

    def computeStaticOutputShape(self, inputNodes):
        """
        compute static output shape(may contain -1)
        should also check if input is valid
        """
        return inputNodes[0].getShape()

    def bulidGradientNodes(self,outputGradientNode,lossNodeName):
        """
        bulid gradient nodes for BP
        return [[input node,corresponding gradient node],...]
        """
        inputNode=self.getInputNodes()[0]
        gradientNode=self.GradientNode(getGradientNodeName(lossNodeName,inputNode.getName()),
                                       [self,outputGradientNode])
        return [[inputNode,gradientNode]]

    class GradientNode(OpNode):
        def compute(self, inputValues, outputBuff):
            """
            warning: numpy array obj is shared in computation,don't modi input value
            :return: output value
            """
            np.power(inputValues[0],2,out=outputBuff)
            np.subtract(1,outputBuff,out=outputBuff)
            outputBuff*=inputValues[1]
            return outputBuff

        def computeStaticOutputShape(self, inputNodes):
            """
            compute static output shape(may contain -1)
            should also check if input is valid
            """
            return inputNodes[0].getShape()