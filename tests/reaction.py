from dataclasses import dataclass


@dataclass(init=True)
class MolInfo:
    filename: str
    geom: str
    spin: int
    charge: int
    n_atoms: int


@dataclass(init=True)
class Educt:
    weight: int
    mol: MolInfo

@dataclass(init=True)
class Product:
    weight: int
    mol: MolInfo

@dataclass(init=True)
class Reaction:
    dataset: str
    file_name: str
    reaction: str
    products: list[Product]
    educts: list[Educt]
    energy_diff_kcalmol: float
    out_weight: int
    elements: set[str]

