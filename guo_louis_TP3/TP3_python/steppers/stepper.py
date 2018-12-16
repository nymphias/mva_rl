from abc import ABC, abstractmethod

class stepper(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def update(self, gt):
        pass