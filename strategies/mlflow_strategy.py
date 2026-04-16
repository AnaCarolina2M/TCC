from abc import ABC, abstractmethod 

class IMLFlowLog(ABC):
    @abstractmethod
    def experiment(self):
        pass