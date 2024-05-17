from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike

from .Aerodynamics import AerodynamicProperties
from .Geometry import BEMGeometry
from .RotorDefinition import RotorDefinition

__all__ = [
    "TangentialInductionModel",
    "NoTangentialInduction",
    "DefaultTangentialInduction",
]


class TangentialInductionModel(ABC):
    @abstractmethod
    def __call__(
        self,
        aero_props: AerodynamicProperties,
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: RotorDefinition,
        geom: BEMGeometry,
    ) -> ArrayLike:
        ...


class NoTangentialInduction(TangentialInductionModel):
    def __call__(
        self,
        aero_props: AerodynamicProperties,
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: RotorDefinition,
        geom: BEMGeometry,
    ) -> ArrayLike:
        return np.zeros_like(aero_props.an)


class DefaultTangentialInduction(TangentialInductionModel):
    def __call__(
        self,
        aero_props: AerodynamicProperties,
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: RotorDefinition,
        geom: BEMGeometry,
    ) -> ArrayLike:
        tangential_integral = aero_props.W**2 * aero_props.Ctan

        aprime = (
            aero_props.solidity
            / (4 * np.maximum(geom.mu_mesh, 0.1) ** 2 * tsr * (1 - aero_props.an) * np.cos(yaw))
            * tangential_integral
        )

        return aprime
