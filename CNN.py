import numpy as np

from baseOps import *
from shapeOps import *
from mathOps import *
from globalFun import *

from nn import *

class PaddingOp(OpNode):
    def __init__(self, name, inputNodes):
        """
        :param inputNodes: [image node, filter size, stride]
                           image node shape is [batch size, width, height, channel]
                           filter size and stride is scalar
        """
        self.leftUp = []
        self.rightBottom = []

        super().__init__(name,inputNodes)

    def compute(self,inputValues,outputBuff):
        """
        warning: numpy array obj is shared in computation,don't modi input value
        :return: output value
        """
        outputBuff[:,self.leftUp[0]:self.rightBottom[0],self.leftUp[1]:self.rightBottom[1],:]=inputValues[0]
        return outputBuff

    def computeStaticOutputShape(self, inputNodes):
        """
        compute static output shape(may contain -1)
        should also check if input is valid
        """
        inputImage = inputNodes[0]
        imageShape = list(inputImage.getShape())

        filterSize = inputNodes[1].getValue()[0]
        stride = inputNodes[2].getValue()[0]

        widths=[imageShape[1],imageShape[2]]
        filterWidths=[filterSize, filterSize]
        strides=[stride, stride]

        paddingWidth=[]

        for i in range(len(widths)):
            width=widths[i]
            filterWidth=filterWidths[i]
            stride=strides[i]

            if width<filterWidth:
                raise Exception("choose a good filter size")

            left=(width-filterWidth)%stride
            if left==0:
                paddingWidth.append(width)
            else:
                paddingWidth.append(width+stride-left)

        for i in range(len(paddingWidth)):
            x1=(paddingWidth[i]-widths[i])//2
            x2=widths[i]+x1

            self.leftUp.append(x1)
            self.rightBottom.append(x2)

        imageShape[1]=paddingWidth[0]
        imageShape[2]=paddingWidth[1]

        return imageShape


    def bulidGradientNodes(self,outputGradientNode,lossNodeName):
        """
        bulid gradient nodes for BP
        return [[input node,corresponding gradient node],...]
        """
        inputNode=self.getInputNodes()[0]
        gradientNode=self.GradientNode(getGradientNodeName(lossNodeName,inputNode.getName()),
                                       outputGradientNode,
                                       inputNode.getShape(),
                                       self.leftUp,
                                       self.rightBottom)
        return [[inputNode,gradientNode]]

    class GradientNode(OpNode):
        def __init__(self,name,gradientNode,outputShape,leftUp,rightBottom):
            self.outputShape=outputShape
            self.leftUp=leftUp
            self.rightBottom=rightBottom

            super().__init__(name,[gradientNode])

        def compute(self, inputValues, outputBuff):
            """
            warning: numpy array obj is shared in computation,don't modi input value
            :return: output value
            """
            outputBuff=inputValues[0][:,self.leftUp[0]:self.rightBottom[0],self.leftUp[1]:self.rightBottom[1],:]
            return outputBuff

        def computeStaticOutputShape(self, inputNodes):
            """
            compute static output shape(may contain -1)
            should also check if input is valid
            """
            return self.outputShape

class Ima2Col(OpNode):
    def __init__(self, name, inputNodes):
        """
        :param inputNodes: [image node, filter size, stride]
                           image node shape is [batch size, width, height, channel]
                           filter size and stride is scalar
        """
        self.filterSize = inputNodes[1][0] #numpy obj
        self.stride = inputNodes[2][0]
        super().__init__(name,inputNodes)

    def compute(self,inputValues,outputBuff):
        """
        warning: numpy array obj is shared in computation,don't modi input value
        :return: output value
        """
        outputShape=self.getShape()

        for x in range(outputShape[1]):
            for y in range(outputShape[2]):
                for c in range(outputShape[3]):
                    x1=x * self.stride
                    y1=y * self.stride

                    data=inputValues[0][:, x1:x1+self.filterSize, y1: y1 + self.filterSize,c]
                    data=np.reshape(data,[-1,self.filterSize*self.filterSize])
                    outputBuff[:,x,y,c,:]=data
        return outputBuff

    def computeStaticOutputShape(self, inputNodes):
        """
        compute static output shape(may contain -1)
        should also check if input is valid
        """
        inputImage = inputNodes[0]
        imageShape = list(inputImage.getShape())

        imageShape[1] = (imageShape[1]-self.filterSize)//self.stride+1
        imageShape[2] = (imageShape[2] - self.filterSize) // self.stride + 1
        imageShape.append(self.filterSize*self.filterSize)

        return imageShape

    def bulidGradientNodes(self,outputGradientNode,lossNodeName):
        """
        bulid gradient nodes for BP
        return [[input node,corresponding gradient node],...]
        """
        inputNode=self.getInputNodes()[0]
        gradientNode=self.GradientNode(getGradientNodeName(lossNodeName,inputNode.getName()),
                                       outputGradientNode,
                                       inputNode.getShape(),
                                       self.filterSize,
                                       self.stride)
        return [[inputNode,gradientNode]]

    class GradientNode(OpNode):
        def __init__(self, name, gradientNode, outputShape, filterSize, stride):
            self.outputShape = outputShape
            self.stride = stride
            self.filterSize = filterSize

            super().__init__(name, [gradientNode])

        def compute(self, inputValues, outputBuff):
            """
            warning: numpy array obj is shared in computation,don't modi input value
            :return: output value
            """
            inputShape = inputValues[0].shape
            filterSize = self.filterSize
            stride = self.stride
            outputBuff *= 0.0

            for x in range(inputShape[1]):
                for y in range(inputShape[2]):
                    for c in range(inputShape[3]):
                        x1 = x * stride
                        y1 = y * stride

                        try:
                            data = inputValues[0][:, x, y, c, :]
                            data = np.reshape(data, [-1, filterSize, filterSize])
                            outputBuff[:, x1:x1 + filterSize, y1: y1 + filterSize, c] += data
                        except Exception as e:
                            raise e

            return outputBuff

        def computeStaticOutputShape(self, inputNodes):
            """
            compute static output shape(may contain -1)
            should also check if input is valid
            """
            return self.outputShape

class ChannelMaxOp(OpNode):
    def compute(self,inputValues,outputBuff):
        """
        warning: numpy array obj is shared in computation,don't modi input value
        :return: output value
        """
        outputBuff=np.reshape(outputBuff,self.getShape()[:-1])
        np.max(inputValues[0],axis=-1,out=outputBuff)
        outputBuff=np.reshape(outputBuff,self.getShape())
        return outputBuff

    def computeStaticOutputShape(self, inputNodes):
        """
        compute static output shape(may contain -1)
        should also check if input is valid
        """
        shape=list(inputNodes[0].getShape())
        shape[-1]=1
        return shape

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
            outputBuff*=0.0
            bpGradient=inputValues[1]
            outputShape=outputBuff.shape

            #[batch,width,height,channel,feature]
            #stupid implement
            for b in range(outputShape[0]):
                for w in range(outputShape[1]):
                    for h in range(outputShape[2]):
                        for c in range(outputShape[3]):
                            outputBuff[b][w][h][c][np.argmax(inputValues[0][b][w][h][c])]=bpGradient[b][w][h][c][0]
            return outputBuff

        def computeStaticOutputShape(self, inputNodes):
            """
            compute static output shape(may contain -1)
            should also check if input is valid
            """
            return inputNodes[0].getShape()

class ChannelAverageOp(OpNode):
    def compute(self,inputValues,outputBuff):
        """
        warning: numpy array obj is shared in computation,don't modi input value
        :return: output value
        """
        outputBuff=np.reshape(outputBuff, self.getShape()[:-1])
        np.mean(inputValues[0],-1,out=outputBuff)
        outputBuff=np.reshape(outputBuff,self.getShape())
        return outputBuff

    def computeStaticOutputShape(self, inputNodes):
        """
        compute static output shape(may contain -1)
        should also check if input is valid
        """
        shape = list(inputNodes[0].getShape())
        shape[-1] = 1
        return shape

    def bulidGradientNodes(self,outputGradientNode,lossNodeName):
        """
        bulid gradient nodes for BP
        return [[input node,corresponding gradient node],...]
        """
        inputNode = self.getInputNodes()[0]
        gradientNode = self.GradientNode(getGradientNodeName(lossNodeName, inputNode.getName()),
                                         [inputNode, outputGradientNode])
        return [[inputNode, gradientNode]]

    class GradientNode(OpNode):
        def __init__(self,name,inputNodes):
            super().__init__(name,inputNodes)
            self.expandMat = np.ones([1, inputNodes[0].getShape()[-1]], np.float32)
            self.expandMat *= 1 / inputNodes[0].getShape()[-1]

        def compute(self, inputValues, outputBuff):
            """
            warning: numpy array obj is shared in computation,don't modi input value
            :return: output value
            """
            np.matmul(inputValues[1],self.expandMat,out=outputBuff)
            return outputBuff

        def computeStaticOutputShape(self, inputNodes):
            """
            compute static output shape(may contain -1)
            should also check if input is valid
            """
            return inputNodes[0].getShape()

def convLayer(input,size,stride,filterNum,name):
    paddedInput=PaddingOp(input.getName()+"_padding",
                          [input,np.array([size]),np.array([stride])])
    colImage=Ima2Col(paddedInput.getName()+"_toCol",
                     [input, np.array([size]), np.array([stride])])

    shape=list(colImage.getShape())
    reshaped=ReshapeOp(colImage.getName()+"_reshape",
                       [colImage,np.array([-1,shape[-1]*shape[-2]])])
    conv=denseLayer(reshaped,filterNum,reshaped.getName()+"_conv")
    convReshape=ReshapeOp(name,[conv,np.array([-1,shape[1],shape[2],filterNum])])

    return convReshape

def poolImage(poolOpClass,poolName,input,size,stride,name):
    paddedInput = PaddingOp(input.getName() + "_padding",
                            [input, np.array([size]), np.array([stride])])
    colImage = Ima2Col(paddedInput.getName() + "_toCol",
                       [paddedInput, np.array([size]), np.array([stride])])
    pooled = poolOpClass(colImage.getName() + poolName, [colImage])
    shape = list(pooled.getShape())
    reshaped = ReshapeOp(name, [pooled, np.array(shape[:-1])])
    return reshaped

#max pool gradient bp is very slow, use conv with stride 2
def maxPool(input,size,stride,name):
    return poolImage(ChannelMaxOp,"_maxPool",input,size,stride,name)

def avgPool(input,size,stride,name):
    return poolImage(ChannelAverageOp, "_avgPool", input, size, stride, name)