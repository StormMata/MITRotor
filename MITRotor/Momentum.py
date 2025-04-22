from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal
import numpy as np
from numpy.typing import ArrayLike
import torch
import torch.nn as nn
import torch.optim as optim

from UnifiedMomentumModel import Momentum as UMM

if TYPE_CHECKING:
    from .Geometry import BEMGeometry
    from .RotorDefinition import RotorDefinition
    from .Aerodynamics import AerodynamicProperties

__all__ = [
    "MomentumModel",
    "ConstantInduction",
    "NeuralNetInduction",
    "ClassicalMomentum",
    "HeckMomentum",
    "UnifiedMomentum",
    "MadsenMomentum",
    "Madsen_Annulus_Momentumm",
    "Madsen_10MWAnnulus_Momentum",
    "Madsen_Rotor_Momentum",
    "Madsen_10MWRotor_Momentum",
]

class XY_Predictor(nn.Module):
    def __init__(self):
        super(XY_Predictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 64),  # Input size changed from 4 → 5
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output size changed from 2 → 1 (predicting x only)
        )

    def forward(self, x):
        return self.model(x)


def evaluate_model(model, r, theta, z1_eval, z2_eval, y_true):
    """Evaluate the model on a specific (z1, z2) case."""
    # Prepare input
    X_eval = np.column_stack([r.flatten(), theta.flatten(),
                              np.full_like(r.flatten(), z1_eval),
                              np.full_like(r.flatten(), z2_eval),
                              y_true.flatten()])  # Add y as input
    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)

    # Get predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X_eval_tensor).numpy()

    # Reshape predictions
    x_pred = predictions[:, 0].reshape(r.shape)

    return x_pred


class MomentumModel(ABC):
    @abstractmethod
    def compute_induction(self, Cx: ArrayLike, yaw: float) -> ArrayLike:
        ...

    @abstractmethod
    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
        a: float,
    ) -> ArrayLike:
        ...

    def _func_annulus(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        
        rotor_avg_axial_force = (
            geom.rotor_average(
                geom.annulus_average(
                    np.clip(aero_props.C_x_corr, 0, 1.69)
                    )
                    )
        )

        return self.compute_induction(rotor_avg_axial_force, yaw)


    def _func_annulus(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        
        annulus_avg_axial_force = (
            
                geom.annulus_average(
                    aero_props.C_x_corr
                    )
                    )[:, None] * np.ones(geom.shape)
        

        return self.compute_induction(annulus_avg_axial_force, yaw)

    def _func_sector(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        axial_force = aero_props.C_x_corr

        return self.compute_induction(axial_force, yaw)

    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
        a: float,
    ) -> ArrayLike:
        an = self._func(aero_props, pitch, tsr, yaw, rotor, geom)
        return np.clip(an, 0, 1)

class ConstantInduction(MomentumModel):
    def __init__(self, a):
        self.a = a

    def _func(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
        a: float,
    ) -> ArrayLike:
        return self.a * np.ones_like(aero_props.an)


class ClassicalMomentum(MomentumModel):
    def __init__(self, averaging: Literal["sector", "annulus", "rotor"] = "rotor"):
        if averaging == "rotor":
            self._func = self._func_rotor
        elif averaging == "annulus":
            self._func = self._func_annulus
        elif averaging == "sector":
            self._func = self._func_sector
        else:
            raise ValueError(f"Averaging method {averaging} not found for ClassicalMomentum model.")
        self.averaging = averaging

    def compute_induction(self, Cx, yaw):
        return 0.5 * (1 - np.sqrt(1 - Cx))

class NeuralNetInduction(MomentumModel):
    def __init__(self, cosine_exponent=None, shear=0, veer=0):
        self.cosine_exponent = cosine_exponent
        self.shear = shear
        self.veer = veer

    def compute_induction(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
        a: float,
    ) -> ArrayLike:
        Ct = aero_props.solidity * aero_props.W**2 * aero_props.C_x

        NN_a_model = XY_Predictor()
        NN_a_model.load_state_dict(torch.load("/scratch/09909/smata/induction_modeling/FF_NN_modeling/NN_models/E128_L3_N64_Arelu_15.pth"))

        an = evaluate_model(NN_a_model, geom.mu_mesh, geom.theta_mesh, self.veer, self.shear, Ct)

        return an

class HeckMomentum(MomentumModel):
    """
    Heck Momentum model based on 2023 paper:
    https://doi.org/10.1017/jfm.2023.129 
    """
    def __init__(
        self, averaging: Literal["sector", "annulus", "rotor"] = "rotor", ac: float = 1 / 3, v4_correction: float = 1.0
    ):
        self.v4_correction = v4_correction
        self.ac = ac
        if averaging == "rotor":
            self._func = self._func_rotor
        elif averaging == "annulus":
            self._func = self._func_annulus
        elif averaging == "sector":
            self._func = self._func_sector
        else:
            raise ValueError(f"Averaging method {averaging} not found for HeckMomentum model.")
        self.averaging = averaging

    def compute_induction(self, Cx: ArrayLike, yaw: float) -> ArrayLike:
        Ctc = 4 * self.ac * (1 - self.ac) / (1 + 0.25 * (1 - self.ac) ** 2 * np.sin(yaw) ** 2)
        slope = (16 * (1 - self.ac) ** 2 * np.sin(yaw) ** 2 - 128 * self.ac + 64) / (
            (1 - self.ac) ** 2 * np.sin(yaw) ** 2 + 4
        ) ** 2

        a = (2 * Cx - 4 + np.sqrt(-(Cx**2) * np.sin(yaw) ** 2 - 16 * Cx + 16)) / (
            -4 + np.sqrt(-(Cx**2) * np.sin(yaw) ** 2 - 16 * Cx + 16)
        )

        if np.iterable(Cx):
            mask = Cx > Ctc
            if np.any(mask):
                a[mask] = (Cx[mask] - Ctc) / slope + self.ac

        return a


class UnifiedMomentum(MomentumModel):
    """
    Unified Momentum Model based on 2024 paper:
    https://www.nature.com/articles/s41467-024-50756-5 
    """
    def __init__(self, averaging: Literal["sector", "annulus", "rotor"] = "rotor", beta=0.1403):
        self.beta = beta

        if averaging == "rotor":
            self._func = self._func_rotor
        elif averaging == "annulus":
            self._func = self._func_annulus
        elif averaging == "sector":
            self._func = self._func_sector
        else:
            raise ValueError(f"Averaging method {averaging} not found for UnifiedMomentum model.")
        self.averaging = averaging

        self.model_Ct = UMM.ThrustBasedUnified(beta=beta)

    def compute_induction(self, Cx: ArrayLike, yaw: float) -> ArrayLike:
        sol = self.model_Ct(Cx, yaw)
        return sol.an


class MadsenMomentum(MomentumModel):
    """
    Madsen Momentum model based on 2020 paper:
    https://wes.copernicus.org/articles/5/1/2020/
    """
    def __init__(self, 
                 averaging: Literal["sector", "annulus", "rotor"] = "rotor",
                 cosine_exponent: bool = False):
        if averaging == "rotor":
            self._func = self._func_rotor
        elif averaging == "annulus":
            self._func = self._func_annulus
        elif averaging == "sector":
            self._func = self._func_sector
        else:
            raise ValueError(f"Averaging method {averaging} not found for MadsenMomentum model.")
        self.averaging = averaging
        self.cosine_exponent = cosine_exponent


    def compute_induction(self, Cx: ArrayLike, yaw: float) -> ArrayLike:
        if self.cosine_exponent:
            Ct = Cx / (np.cos(yaw)**2)
        else:
            Ct = Cx

        an = Ct**3 * 0.0883 + Ct**2 * 0.0586 + Ct * 0.2460
        return an
    



class Madsen_Annulus_Momentum(MomentumModel):
    def __init__(self, cosine_exponent=None, veer=0):
        self.cosine_exponent = cosine_exponent
        self.veer  = veer

    def Ct_a(self, Ct: ArrayLike,) -> ArrayLike:
        Ct = np.clip(Ct, 0.0, 1.44)

        if int(abs(self.veer)) == 0:
            an = 0.184 * (Ct**3) + -0.128 * (Ct**2) + 0.271 * Ct +0.05
        elif int(abs(self.veer)) == 2:
            an = Ct**3 * 0.166 + Ct**2 * -0.139+ Ct * 0.347 +0.05
        elif int(abs(self.veer)) == 4:
            an = Ct**3 * 0.129 + Ct**2 * -0.099 + Ct * 0.394 +0.05
        else:
            raise ValueError(f"Unsupported veer: {self.veer}")

        return an

    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
        a: float,
    ) -> ArrayLike:

        Ct = geom.annulus_average(aero_props.solidity * aero_props.W**2 * aero_props.Cax)

        an = self.Ct_a(Ct)[:, None] * np.ones(geom.shape)

        return an

class Madsen_10MWAnnulus_Momentum(MomentumModel):
    def __init__(self, cosine_exponent=None, veer=0):
        self.cosine_exponent = cosine_exponent
        self.veer  = veer

    def Ct_a(self, Ct: ArrayLike,) -> ArrayLike:
        Ct = np.clip(Ct, 0.0, 1.44)

        if int(abs(self.veer)) == 0:
            an = 0.184 * (Ct**3) + -0.149 * (Ct**2) + 0.308 * Ct
        elif int(abs(self.veer)) == 2:
            an = Ct**3 * 0.172 + Ct**2 * -0.170+ Ct * 0.383
        elif int(abs(self.veer)) == 4:
            an = Ct**3 * 0.127 + Ct**2 * -0.101 + Ct * 0.407
        else:
            raise ValueError(f"Unsupported veer: {self.veer}")

        return an

    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
        a: float,
    ) -> ArrayLike:

        Ct = geom.annulus_average(aero_props.solidity * aero_props.W**2 * aero_props.Cax)

        an = self.Ct_a(Ct)[:, None] * np.ones(geom.shape)

        return an

class Madsen_Rotor_Momentum(MomentumModel):
    def __init__(self, cosine_exponent=None, veer=0):
        self.cosine_exponent = cosine_exponent
        self.veer  = veer

    def Ct_a(self, Ct: ArrayLike,) -> ArrayLike:
        Ct = np.clip(Ct, 0.0, 1.44)

        if int(abs(self.veer)) == 0:
            an = 0.139 * (Ct**3) + -0.063 * (Ct**2) +  0.268 * Ct 
        elif int(abs(self.veer)) == 2:
            an = Ct**3 * 0.083 + Ct**2 * 0.069+ Ct * 0.244
        elif int(abs(self.veer)) == 4:
            an = Ct**3 * 0.034 + Ct**2 * 0.248 + Ct * 0.201
        else:
            raise ValueError(f"Unsupported veer: {self.veer}")

        return an

    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
        a: float,
    ) -> ArrayLike:

        Ct = geom.annulus_average(aero_props.solidity * aero_props.W**2 * aero_props.Cax)

        an = self.Ct_a(Ct)[:, None] * np.ones(geom.shape)

        return an

class Madsen_10MWRotor_Momentum(MomentumModel):
    def __init__(self, cosine_exponent=None, veer=0):
        self.cosine_exponent = cosine_exponent
        self.veer  = veer

    def Ct_a(self, Ct: ArrayLike,) -> ArrayLike:
        Ct = np.clip(Ct, 0.0, 1.44)

        if int(abs(self.veer)) == 0:
            an = 0.139 * (Ct**3) + -0.063 * (Ct**2) +  0.268 * Ct 
        elif int(abs(self.veer)) == 2:
            an = Ct**3 * 0.083 + Ct**2 * 0.069+ Ct * 0.244
        elif int(abs(self.veer)) == 4:
            an = Ct**3 * 0.034 + Ct**2 * 0.248 + Ct * 0.201
        else:
            raise ValueError(f"Unsupported veer: {self.veer}")

        return an

    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
        a: float,
    ) -> ArrayLike:

        Ct = geom.annulus_average(aero_props.solidity * aero_props.W**2 * aero_props.Cax)

        an = self.Ct_a(Ct)[:, None] * np.ones(geom.shape)

        return an
