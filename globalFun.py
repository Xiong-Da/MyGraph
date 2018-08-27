import numpy as np

nodeMap={}
feed_dic={} #{value node name:value}
nodeList=[]  #store node by construct sequence

executeID=0

def checkName(name):
    if name in nodeMap.keys():
        return False
    return True

def registName(name,node):
    if name not in nodeMap.keys():
        nodeMap[name]=node
        nodeList.append(node)
    else:
        raise Exception("name conflict:"+name)

def getNodeByConstructSeq():
    return nodeList[:]

def getNodeByName(name):
    if name in nodeMap.keys():
        return nodeMap[name]
    return None

def updateExecuteID():
    global executeID
    executeID+=1

def getExecuteID():
    return executeID

def setFeedDic(dataMap):
    global feed_dic
    feed_dic=dataMap.copy()
    updateExecuteID()

def getFeedvalue(name):
    return np.array(feed_dic[name])
