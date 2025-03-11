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

class Predictor(nn.Module):
    def __init__(self, input_size=5, hidden_sizes=[16], output_size=1):
        """
        Generic predictor model with configurable hidden layers.
        
        :param input_size: Number of input features.
        :param hidden_sizes: List of hidden layer sizes.
        :param output_size: Number of output features.
        """
        super(Predictor, self).__init__()
        
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.model = nn.Sequential(*layers)

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
    def Ct_a(self, Ct: ArrayLike, yaw: float) -> ArrayLike:
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


class ConstantInduction(MomentumModel):
    def __init__(self, a):
        self.a = a

    def Ct_a(self, Ct: ArrayLike, yaw: float) -> ArrayLike:
        return self.a * np.ones_like(Ct)

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
        return self.a * np.ones_like(aero_props.an)


class ClassicalMomentum(MomentumModel):
    def Ct_a(self, Ct, yaw, tiploss=1.0):
        return 0.5 * (1 - np.sqrt(1 - Ct / tiploss))

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
        Ct = geom.rotor_average(geom.annulus_average(aero_props.solidity * aero_props.W**2 * aero_props.Cax))

        a = self.Ct_a(Ct, yaw, tiploss=1.0)

        return a

class NeuralNetInduction(MomentumModel):
    def __init__(self, cosine_exponent=None, shear=0, veer=0):
        self.cosine_exponent = cosine_exponent
        self.shear = shear
        self.veer = veer

    def Ct_a(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        Ct = aero_props.solidity * aero_props.W**2 * aero_props.Cax

        # Predictor(hidden_sizes=[32])     # 2 layers, 32 neurons in the hidden layer
        # Predictor(hidden_sizes=[16, 16]) # 3 layers, 16 neurons in each hidden layer

        NN_a_model = Predictor(hidden_sizes=[32])
        NN_a_model.load_state_dict(torch.load("/scratch/09909/smata/induction_modeling/FF_NN_modeling/NN_models/E064_L2_N16_Arelu.pth"))

        an = evaluate_model(NN_a_model, geom.mu_mesh, geom.theta_mesh, self.veer, self.shear, Ct)

        return an

    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        Ct = aero_props.solidity * aero_props.W**2 * aero_props.Cax

        NN_a_model = Predictor(hidden_sizes=[32])
        NN_a_model.load_state_dict(torch.load("/scratch/09909/smata/induction_modeling/FF_NN_modeling/NN_models/E064_L2_N16_Arelu.pth"))

        an = evaluate_model(NN_a_model, geom.mu_mesh, geom.theta_mesh, self.veer, self.shear, Ct)

        return an


class HeckMomentum(MomentumModel):
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

    def Ct_a(self, Ct: ArrayLike, yaw: float) -> ArrayLike:
        Ctc = 4 * self.ac * (1 - self.ac) / (1 + 0.25 * (1 - self.ac) ** 2 * np.sin(yaw) ** 2)
        slope = (16 * (1 - self.ac) ** 2 * np.sin(yaw) ** 2 - 128 * self.ac + 64) / (
            (1 - self.ac) ** 2 * np.sin(yaw) ** 2 + 4
        ) ** 2

        a_target = (2 * Ct - 4 + np.sqrt(-(Ct**2) * np.sin(yaw) ** 2 - 16 * Ct + 16)) / (
            -4 + np.sqrt(-(Ct**2) * np.sin(yaw) ** 2 - 16 * Ct + 16)
        )

        if np.iterable(Ct):
            mask = Ct > Ctc
            if np.any(mask):
                a_target[mask] = (Ct[mask] - Ctc) / slope + self.ac

        return a_target

    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        an = self._func(aero_props, pitch, tsr, yaw, rotor, geom)
        return an

    def _func_rotor(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        Ct = aero_props.solidity * aero_props.W**2 * aero_props.Cax
        Ct_rotor = geom.rotor_average(geom.annulus_average(Ct))

        a_target = self.Ct_a(Ct_rotor, yaw)

        a_new = aero_props.F
        a_rotor = geom.rotor_average(geom.annulus_average(a_new))
        a_new *= a_target / a_rotor

        return a_new

    def _func_annulus(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        Ct = geom.annulus_average(aero_props.solidity * aero_props.W**2 * aero_props.Cax)
        _Ct = np.clip(Ct, -1, 1.59)
        a = self.Ct_a(_Ct, yaw)[:, None] * np.ones(geom.shape)

        return a

    def _func_sector(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        Ct = aero_props.solidity * aero_props.W**2 * aero_props.Cax
        ans = self.Ct_a(Ct.ravel(), yaw)
        return ans.reshape(geom.shape)


class UnifiedMomentum(MomentumModel):
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

        self.model_Ctprime = UMM.UnifiedMomentum(beta=beta)
        self.model_Ct = UMM.ThrustBasedUnified(beta=beta)

    def Ct_a(self, Ct: ArrayLike, yaw: float) -> ArrayLike:
        sol = self.model_Ct(Ct, yaw)
        return sol.an

    def _func_rotor(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        Ct = geom.annulus_average(aero_props.solidity * aero_props.W**2 * aero_props.Cax)

        Ct_rotor = geom.rotor_average(Ct)
        sol = self.model_Ct(Ct_rotor, yaw)
        a_target = sol.an

        a_new = aero_props.F
        a_rotor = geom.rotor_average(geom.annulus_average(a_new))
        a_new *= a_target / a_rotor

        return a_new

    def _func_annulus(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        Ct = geom.annulus_average(aero_props.solidity * aero_props.W**2 * aero_props.Cax)
        _Ct = np.clip(Ct, -1, 1.59)
        an = self.Ct_a(_Ct, yaw)[:, None] * np.ones(geom.shape)
        return an

    def _func_sector(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        Ct = aero_props.solidity * aero_props.W**2 * aero_props.Cax
        an = self.Ct_a(Ct.ravel(), yaw).reshape(geom.shape)
        return an

    def __call__(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:
        an = self._func(aero_props, pitch, tsr, yaw, rotor, geom)
        return an


class MadsenMomentum(MomentumModel):
    def __init__(self, cosine_exponent=None):
        self.cosine_exponent = cosine_exponent

    def Ct_a(self, Ct: ArrayLike, yaw: float, tiploss=1.0) -> ArrayLike:
        Ct_tiploss = np.clip(Ct / tiploss, 0.0, 2.0)
        if self.cosine_exponent:
            Ct_tiploss /= np.cos(yaw) ** self.cosine_exponent
        an = Ct_tiploss**3 * 0.0883 + Ct_tiploss**2 * 0.0586 + Ct_tiploss * 0.2460
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
        Ct = aero_props.solidity * aero_props.W**2 * aero_props.Cax
        an = self.Ct_a(Ct, yaw, tiploss=aero_props.F)
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