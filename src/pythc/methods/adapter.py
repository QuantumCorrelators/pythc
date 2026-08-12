from abc import ABC, abstractmethod

from pyscf import gto
from pyscf.scf.hf import SCF

from pythc.configurable import Configurable
from pythc.thc.thc_base import ThcEri

class Adapter(ABC,Configurable):
    @abstractmethod
    def compare(self, mol: gto.Mole, mf: SCF, eri: ThcEri):
        pass

    @classmethod
    @abstractmethod
    def need_eri(cls) -> str:
        pass

