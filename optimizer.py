import numpy as np

from baseOps import *
from shapeOps import *
from globalFun import *

def countGradientNode(lossNode,countMap):
    if lossNode.getName() in list(countMap.keys()):
        countMap[lossNode.getName()]+=1
        return
    countMap[lossNode.getName()]=1

    parentOp=lossNode.getParentOp()
    if parentOp==None:
        return

    for node in parentOp.getInputNodes():
        countGradientNode(node,countMap)

def minimize(lossNode):
    """
    return [[weightNode,weightGradientNode],...]
    """
    if lossNode.getShape() != (1,):
        raise Exception("only support scalar node")

    lossNodeName=lossNode.getName()

    nodeGradientNodeCountMap = {}
    countGradientNode(lossNode, nodeGradientNodeCountMap)

    inProcessNodesGradientMap={}
    inProcessOpList=[]
    nodeGradientPareList=[]

    parentOp=lossNode.getParentOp()
    if parentOp==None:
        return []
    initGradientNode=ConstValueNode("init gradient node",np.array([1.0],np.float32))

    inProcessOpList.append([parentOp,initGradientNode])

    while True:
        if len(inProcessOpList)==0:
            break

        for op,bpGradient in inProcessOpList:
            for node,gradientNode in op.bulidGradientNodes(bpGradient,lossNodeName):
                nodeName=node.getName()
                if nodeName not in inProcessNodesGradientMap.keys():
                    inProcessNodesGradientMap[nodeName]=[]
                inProcessNodesGradientMap[nodeName].append(gradientNode)

        inProcessOpList=[]

        for nodeName in list(inProcessNodesGradientMap.keys()):
            if len(inProcessNodesGradientMap[nodeName])==nodeGradientNodeCountMap[nodeName]:
                node=getNodeByName(nodeName)
                gradientNodes=inProcessNodesGradientMap[nodeName]
                if len(gradientNodes)==1:
                    finalGradentNode=gradientNodes[0]
                else:
                    finalGradentNode = AddOp(lossNodeName+"@"+nodeName+"_gradientSum",
                                             gradientNodes)
                nodeGradientPareList.append([node,finalGradentNode])

                inProcessNodesGradientMap.pop(nodeName)
                parentOp=node.getParentOp()
                if parentOp!=None:
                    inProcessOpList.append([parentOp,finalGradentNode])

    if len(list(inProcessNodesGradientMap.keys()))!=0:
        raise Exception("implement error, this map should be empty")

    returnList=[]
    for node,gradientNode in nodeGradientPareList:
        if type(node)==WeightValueNode:
            returnList.append([node,gradientNode])
    return returnList

class SGDNode:
    def __init__(self,name,nodeGradientPareList,learningRateNode):
        self.name=name
        self.nodeGradientPareList=nodeGradientPareList[:]
        self.learningRateNode=learningRateNode

    def getValue(self):
        self.updateWeight(self.computeUpdate(self.computGradient()))
        return None

    def computGradient(self):
        nodeGredientPairs = []
        for node, gradientNode in self.nodeGradientPareList:
            gradient = gradientNode.getValue()
            #gradient = np.maximum(np.minimum(gradient,0.1),-0.1)
            nodeGredientPairs.append([node, gradient])
        return nodeGredientPairs

    def computeUpdate(self,nodeGredientPairs):
        learningRate=-1*self.learningRateNode.getValue()[0]
        nodeUpdatePairs=[]
        for node,gradient in nodeGredientPairs:
            update=gradient*learningRate
            nodeUpdatePairs.append([node,update])
        return nodeUpdatePairs

    def updateWeight(self,nodeUpdatePairs):
        for node,updateValue in nodeUpdatePairs:
            node.updateValue(updateValue)

    def getName(self):
        return self.name

    def getShape(self):
        return None

class MomentumSGDNode(SGDNode):
    def __init__(self,name,nodeGradientPareList,learningRateNode,momentumNode):
        super().__init__(name,nodeGradientPareList,learningRateNode)
        self.momentumNode=momentumNode

        self.updateMap={}
        for node,gradientNode in nodeGradientPareList:
            self.updateMap[node.getName()]=np.zeros(node.getShape(),np.float32)

    def computeUpdate(self,nodeGredientPairs):
        momentum=self.momentumNode.getValue()[0]
        learningRate = -1 * self.learningRateNode.getValue()[0]
        nodeUpdatePairs = []

        for node, gradient in nodeGredientPairs:
            update = self.updateMap[node.getName()]
            update*=momentum
            update+=gradient*learningRate

            nodeUpdatePairs.append([node, update])
            self.updateMap[node.getName()]=update

        return nodeUpdatePairs