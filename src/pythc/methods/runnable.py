from abc import ABC, abstractmethod

from pythc.configurable import Configurable

class Runnable(ABC,Configurable):
    @abstractmethod
    def kernel(self) -> tuple[float,...]:
        pass
