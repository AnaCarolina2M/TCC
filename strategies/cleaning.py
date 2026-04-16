from abc import ABC, abstractmethod

class IDataClean(ABC):
    @abstractmethod
    def cleans(self):
        pass