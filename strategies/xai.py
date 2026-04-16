from abc import ABC, abstractmethod

class IXAI(ABC):
    @abstractmethod
    def explains(self):
        pass