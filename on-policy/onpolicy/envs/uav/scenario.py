import numpy as np

# defines scenario upon which the world is built
class BaseScenarioUAV(object):
    def __init__(self):
        self.map = dict()
        self.map["position"] = {"dim": 3, "inx": [i for i in range(0, 3)]}
        self.map["orientation"] = {"dim": 3, "inx": [i for i in range(3,6)]}
        self.map["velocity"] = {"dim": 3, "inx": [i for i in range(6,9)]}
        self.map["bodyrate"] = {"dim": 3, "inx": [i for i in range(9,12)]}
        #self.map["fuel"] = {"dim": 1, "inx": [12,13]}
        self.map["fuel"] = {"dim": 1, "inx": [12]}

    def get_observation(self):
        raise NotImplementedError 
    
    def getReward(self):
        raise NotImplementedError 
    
    def getDone(self):
        raise NotImplementedError 
    
    def getInfo(self):
        raise NotImplementedError 