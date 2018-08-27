import numpy as np

from globalFun import *

class ValueNode():
    def __init__(self,name,shape):
        """
        :param name: a str unique to other node
        :param shape: list,[] means scalar
        """
        registName(name,self)
        self.name=name
        self.shape=tuple(shape) #the first dimention of shape can be -1
        self.computedValue=None
        self.valueEdition=None

        self.opsReferencceThisNode=[] #recorde ops use this node as input

    def getName(self):
        return self.name

    def getShape(self):
        return self.shape

    def getValue(self):
        """
        avoid compute a node muti time in a execution
        """
        if self.valueEdition!=getExecuteID():
            self.valueEdition=getExecuteID()
            self.computedValue=self.__compute__()
        return self.computedValue

    def setReferenceOp(self,op):
        self.opsReferencceThisNode.append(op)

    def getReferenceOp(self):
        return self.opsReferencceThisNode[:]

    def getParentOp(self):
        """
        return op generate this node
        """
        return None

    def __compute__(self):
        """
        sub class should implement this function
        """
        raise Exception("getValue fun unimplement")

class ConstValueNode(ValueNode):
    def __init__(self,name,value):
        self.value = np.array(value)
        super().__init__(name, value.shape)

    def __compute__(self):
        return self.value

class WeightValueNode(ValueNode):
    def __init__(self,name,value):
        self.value = np.array(value,np.float32)
        super().__init__(name, value.shape)

    def __compute__(self):
        return self.value

    def updateValue(self,update):
        self.value+=update # += will not create new array

class Placeholder(ValueNode):
    def __init__(self,name,shape):
        super().__init__(name,shape)

    def __compute__(self):
        value=getFeedvalue(self.getName())

        defineShape=self.getShape()
        givenShape=value.shape

        if len(defineShape)!=len(givenShape):
            raise Exception("invalid feed data for placeholder:"+self.getName())

        for i in range(len(defineShape)):
            if defineShape[i]!=givenShape[i] and defineShape[i]!=-1:
                raise Exception("invalid feed data for placeholder:"+self.getName())
        return value