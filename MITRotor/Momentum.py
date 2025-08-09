from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal
import numpy as np
from numpy.typing import ArrayLike
import torch
import torch.nn as nn
import joblib
import pickle
import os
import pandas as pd
from UnifiedMomentumModel import Momentum as UMM

if TYPE_CHECKING:
    from .Geometry import BEMGeometry
    from .RotorDefinition import RotorDefinition
    from .Aerodynamics import AerodynamicProperties

__all__ = [
    "MomentumModel",
    "ConstantInduction",
    "ClassicalMomentum",
    "HeckMomentum",
    "UnifiedMomentum",
    "MadsenMomentum",
    "Madsen_Annulus_Momentumm",
    "Madsen_10MWAnnulus_Momentum",
    "Madsen_Rotor_Momentum",
    "Madsen_10MWRotor_Momentum",
    "GP_Rotor",
    "GP_Annulus"
]

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

    def _func_rotor(
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

        return self.compute_induction(aero_props, geom)


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
        

        return self.compute_induction(aero_props, geom)

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

        return self.compute_induction(geom)

    def _func_NN_sector(
        self,
        aero_props: "AerodynamicProperties",
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: "RotorDefinition",
        geom: "BEMGeometry",
    ) -> ArrayLike:

        return self.compute_induction(aero_props, geom, pitch, tsr, yaw)

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
        self._func = self._func_sector

    def compute_induction(
        self,
        geom: "BEMGeometry",
    ) -> ArrayLike:
        # Ct = aero_props.solidity * aero_props.W**2 * aero_props.C_x

        return self.a * np.ones_like(geom.mu_mesh)


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

    def compute_induction(self, aero_props, geom):
        # return 0.5 * (1 - np.sqrt(1 - aero_props.C_x))
        Ct = geom.rotor_average(geom.annulus_average(np.clip(aero_props.C_x, 0.0, 1.69)))
        return 0.5 * (1 - np.sqrt(1 - Ct))
    

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


    def compute_induction(self, aero_props, geom) -> ArrayLike:
        # if self.cosine_exponent:
        #     Ct = aero_props.C_x / (np.cos(yaw)**2)
        # else:
        Ct = aero_props.C_x

        Ct = geom.rotor_average(geom.annulus_average(np.clip(Ct, 0.0, 1.69)))

        an = Ct**3 * 0.0883 + Ct**2 * 0.0586 + Ct * 0.2460
        return an

class Madsen_Rotor_Momentum_PosV_NoS(MomentumModel):
    def __init__(self, veer):
        self.veer  = veer
        self._func = self._func_rotor

    def compute_induction(self, aero_props, geom) -> ArrayLike:

        Ct = aero_props.C_x

        Ct = geom.rotor_average(geom.annulus_average(np.clip(Ct, 0.0, 1.69)))
        
        if int(self.veer) == 8:
            a,b,c = 0.222, 0.014, 0.253
        elif int(self.veer) == 7:
            a,b,c = 0.206, 0.018, 0.253
        elif int(self.veer) == 6:
            a,b,c = 0.186, 0.023, 0.252
        elif int(self.veer) == 5:
            a,b,c = 0.165, 0.028, 0.252
        elif int(self.veer) == 4:
            a,b,c = 0.145, 0.034, 0.252
        elif int(self.veer) == 3:
            a,b,c = 0.126, 0.039, 0.251
        elif int(self.veer) == 2:
            a,b,c = 0.110, 0.043, 0.251
        elif int(self.veer) == 1:
            a,b,c = 0.099, 0.046, 0.251
        elif int(self.veer) == 0:
            a,b,c = 0.093, 0.048, 0.251
        else:
            raise ValueError(f"Unsupported veer: {self.veer}")

        return a * Ct**3 + b * Ct**2 + c * Ct
    
class Madsen_Rotor_Momentum_NegV_NoS(MomentumModel):
    def __init__(self, veer):
        self.veer  = veer
        self._func = self._func_rotor

    def compute_induction(self, aero_props, geom) -> ArrayLike:

        Ct = aero_props.C_x_corr

        Ct = geom.rotor_average(geom.annulus_average(np.clip(Ct, 0.0, 1.69)))
        # theta = np.linspace(0.0, 2 * np.pi, 158)
        # mu = np.linspace(0, 0.99999, 26)

        # Ct = 2 * np.trapezoid(1/(2 * np.pi) * np.trapezoid(np.clip(Ct, 0.0, 1.69), theta, axis=-1) * mu, mu)
        
        # theta = np.linspace(0, 2*np.pi,158)
        # mu = np.array([0.04288751, 0.08042134, 0.11795516, 0.15548898, 0.19302281,0.23055663, 0.26809045, 0.30562428, 0.3431581 , 0.38069192,0.41822574, 0.45575957, 0.49329339, 0.53082721, 0.56836104,0.60589486, 0.64342868, 0.6809625 , 0.71849633, 0.75603015,0.79356397, 0.8310978 , 0.86863162, 0.90616544, 0.94369927,0.98123309])

        # Ct = 2 * np.trapezoid(1/(2 * np.pi) * np.trapezoid(aero_props.C_x, theta, axis=-1) * mu, mu)

        # print(f'CT is: {Ct} \n')

        # if int(self.veer) == -8:
        #     a,b,c = 0.167, 0.028, 0.252
        # elif int(self.veer) == -7:
        #     a,b,c = 0.156, 0.031, 0.252
        # elif int(self.veer) == -6:
        #     a,b,c = 0.143, 0.034, 0.252
        # elif int(self.veer) == -5:
        #     a,b,c = 0.129, 0.038, 0.251
        # elif int(self.veer) == -4:
        # a,b,c = 0.116, 0.041, 0.251
        # elif int(self.veer) == -3:
        #     a,b,c = 0.105, 0.044, 0.251
        # elif int(self.veer) == -2:
        #     a,b,c = 0.097, 0.047, 0.251
        # elif int(self.veer) == -1:
        #     a,b,c = 0.092, 0.048, 0.251
        # elif int(self.veer) == 0:
        # a,b,c = 0.09334, 0.04743, 0.25086
        if int(self.veer) == -8:
            a,b,c = 0.1580, 0.0304, 0.2519
        elif int(self.veer) == -7:
            a,b,c = 0.1476, 0.0332, 0.2517
        elif int(self.veer) == -6:
            a,b,c = 0.1353, 0.0364, 0.2515
        elif int(self.veer) == -5:
            a,b,c = 0.1222, 0.0398, 0.2513
        elif int(self.veer) == -4:
            a,b,c = 0.1094, 0.0432, 0.2511
        elif int(self.veer) == -3:
            a,b,c = 0.0986, 0.0460, 0.2509
        elif int(self.veer) == -2:
            a,b,c = 0.0909, 0.0481, 0.2508
        elif int(self.veer) == -1:
            a,b,c = 0.0869, 0.0491, 0.2508
        elif int(self.veer) == 0:
            a,b,c = 0.0875, 0.0490, 0.2508
        else:
            raise ValueError(f"Unsupported veer: {self.veer}")

        # print(f'a is: {a * Ct**3 + b * Ct**2 + c * Ct } \n')
        return a * Ct**3 + b * Ct**2 + c * Ct 
        # return 1/3

class Madsen_Rotor_Momentum_AllV_NoS(MomentumModel):
    def __init__(self, veer):
        self.veer  = veer
        self._func = self._func_rotor

    def compute_induction(self, aero_props, geom) -> ArrayLike:

        Ct = aero_props.C_x

        Ct = geom.rotor_average(geom.annulus_average(np.clip(Ct, 0.0, 1.69)))
        
        if int(self.veer) == -8:
            a,b,c = 0.167, 0.028, 0.252
        elif int(self.veer) == -7:
            a,b,c = 0.156, 0.031, 0.252
        elif int(self.veer) == -6:
            a,b,c = 0.143, 0.034, 0.252
        elif int(self.veer) == -5:
            a,b,c = 0.129, 0.038, 0.251
        elif int(self.veer) == -4:
            a,b,c = 0.116, 0.041, 0.251
        elif int(self.veer) == -3:
            a,b,c = 0.105, 0.044, 0.251
        elif int(self.veer) == -2:
            a,b,c = 0.097, 0.047, 0.251
        elif int(self.veer) == -1:
            a,b,c = 0.092, 0.048, 0.251
        elif int(self.veer) == 0:
            a,b,c = 0.093, 0.048, 0.251
        elif int(self.veer) == 1:
            a,b,c = 0.099, 0.046, 0.251
        elif int(self.veer) == 2:
            a,b,c = 0.110, 0.043, 0.251
        elif int(self.veer) == 3:
            a,b,c = 0.126, 0.039, 0.251
        elif int(self.veer) == 4:
            a,b,c = 0.145, 0.034, 0.252
        elif int(self.veer) == 5:
            a,b,c = 0.165, 0.028, 0.252
        elif int(self.veer) == 6:
            a,b,c = 0.186, 0.023, 0.252
        elif int(self.veer) == 7:
            a,b,c = 0.206, 0.018, 0.253
        if int(self.veer) == 8:
            a,b,c = 0.222, 0.014, 0.253
        else:
            raise ValueError(f"Unsupported veer: {self.veer}")

        return a * Ct**3 + b * Ct**2 + c * Ct

class Madsen_Rotor_Momentum_AbsV_NoS(MomentumModel):
    def __init__(self, veer):
        self.veer  = veer
        self._func = self._func_rotor

    def compute_induction(self, aero_props, geom) -> ArrayLike:

        Ct = aero_props.C_x

        Ct = geom.rotor_average(geom.annulus_average(np.clip(Ct, 0.0, 1.69)))
        
        if abs(int(self.veer)) == 8:
            a,b,c = 1.181, -0.722, 0.250
        elif abs(int(self.veer)) == 7:
            a,b,c = 1.201, -0.746, 0.250
        elif abs(int(self.veer)) == 6:
            a,b,c = 1.192, -0.750, 0.250
        elif abs(int(self.veer)) == 5:
            a,b,c = 1.156, -0.735, 0.250
        elif abs(int(self.veer)) == 4:
            a,b,c = 1.101, -0.705, 0.250
        elif abs(int(self.veer)) == 3:
            a,b,c = 1.038, -0.668, 0.250
        elif abs(int(self.veer)) == 2:
            a,b,c = 0.977, -0.628, 0.250
        elif abs(int(self.veer)) == 1:
            a,b,c = 0.925, -0.594, 0.250
        elif abs(int(self.veer)) == 0:
            a,b,c = 0.118, 0.023, 0.254
        else:
            raise ValueError(f"Unsupported veer: {self.veer}")

        return a * Ct**3 + b * Ct**2 + c * Ct
    
class Madsen_Rotor_Momentum_AbsV_AllS(MomentumModel):
    def __init__(self, veer):
        self.veer  = veer
        self._func = self._func_rotor

    def compute_induction(self, aero_props, geom) -> ArrayLike:

        Ct = aero_props.C_x

        # Ct = geom.rotor_average(geom.annulus_average(np.clip(Ct, 0.0, 1.69)))

        theta = np.linspace(0, 2*np.pi,158)
        mu = np.array([0.04288751, 0.08042134, 0.11795516, 0.15548898, 0.19302281,0.23055663, 0.26809045, 0.30562428, 0.3431581 , 0.38069192,0.41822574, 0.45575957, 0.49329339, 0.53082721, 0.56836104,0.60589486, 0.64342868, 0.6809625 , 0.71849633, 0.75603015,0.79356397, 0.8310978 , 0.86863162, 0.90616544, 0.94369927,0.98123309])

        Ct = 2 * np.trapezoid(1/(2 * np.pi) * np.trapezoid(aero_props.C_x, theta, axis=-1) * mu, mu)

        print(f'CT is: {Ct} \n')
        
        if abs(int(self.veer)) == 8:
            a,b,c = 1.366, -1.094, 0.422
        elif abs(int(self.veer)) == 7:
            a,b,c = 1.395, -1.134, 0.428
        elif abs(int(self.veer)) == 6:
            a,b,c = 1.368, -1.125, 0.427
        elif abs(int(self.veer)) == 5:
            a,b,c = 0.968, -0.721, 0.343
        elif abs(int(self.veer)) == 4:
            a,b,c = 1.014, -0.822, 0.383
        elif abs(int(self.veer)) == 3:
            a,b,c = 0.428, -0.273, 0.300
        elif abs(int(self.veer)) == 2:
            a,b,c = 2.334, -2.507, 0.880
        elif abs(int(self.veer)) == 1:
            a,b,c = -0.185, 0.323, 0.196
        elif abs(int(self.veer)) == 0:
            a,b,c = -0.314, 0.434, 0.185
        else:
            raise ValueError(f"Unsupported veer: {self.veer}")
        
        # print(a * Ct**3 + b * Ct**2 + c * Ct)

        return a * Ct**3 + b * Ct**2 + c * Ct


class Madsen_Rotor_Momentum_AllV_AllS(MomentumModel):
    def __init__(self, veer):
        self.veer  = veer
        self._func = self._func_rotor

    def compute_induction(self, aero_props, geom) -> ArrayLike:

        Ct = aero_props.C_x

        Ct = geom.rotor_average(geom.annulus_average(Ct))
        
        a_coeffs = np.load('/scratch/09909/smata/induction_modeling/madsen_modeling/rotorAvg_10MW/processedData/a_coeffs.npy')
        b_coeffs = np.load('/scratch/09909/smata/induction_modeling/madsen_modeling/rotorAvg_10MW/processedData/b_coeffs.npy')
        c_coeffs = np.load('/scratch/09909/smata/induction_modeling/madsen_modeling/rotorAvg_10MW/processedData/c_coeffs.npy')

        a = a_coeffs[self.veer[0],self.veer[1]] * Ct**3 + b_coeffs[self.veer[0],self.veer[1]] * Ct**2 + c_coeffs[self.veer[0],self.veer[1]] * Ct

        # print(a_coeffs[self.veer[0],self.veer[1]])

        return a
    

class Madsen_Annulus_Momentum_AllV_AllS(MomentumModel):
    def __init__(self, veer):
        self.veer  = veer
        self._func = self._func_annulus

    def compute_induction(self, aero_props, geom) -> ArrayLike:

        Ct = aero_props.C_x

        Ct = geom.annulus_average(Ct)

        # print(Ct)
        
        a_coeffs = np.load('/scratch/09909/smata/induction_modeling/madsen_modeling/rotorAvg_10MW/processedData/a_ann_coeffs.npy')
        b_coeffs = np.load('/scratch/09909/smata/induction_modeling/madsen_modeling/rotorAvg_10MW/processedData/b_ann_coeffs.npy')
        c_coeffs = np.load('/scratch/09909/smata/induction_modeling/madsen_modeling/rotorAvg_10MW/processedData/c_ann_coeffs.npy')

        a = a_coeffs[self.veer[0],self.veer[1]] * Ct**3 + b_coeffs[self.veer[0],self.veer[1]] * Ct**2 + c_coeffs[self.veer[0],self.veer[1]] * Ct

        # print(a_coeffs[self.veer[0],self.veer[1]])

        # a = np.ones((26,158)).T * a

        # a = a.reshape(-1, 1)

        return a

class Madsen_committee(MomentumModel):
    def __init__(self, veer):
        self.veer  = veer
        self._func = self._func_rotor

    def compute_induction(self, aero_props, geom) -> ArrayLike:

        Ct = aero_props.C_x

        Ct = geom.rotor_average(geom.annulus_average(np.clip(Ct, 0.0, 1.69)))
        
        if int(self.veer) == -8:
            a,b,c = 0.1782, 0.0251, 0.2522
        elif int(self.veer) == -4:
            a,b,c = 0.1278, 0.0384, 0.2514
        elif int(self.veer) == -2:
            a,b,c = 0.1084, 0.0435, 0.2511
        elif int(self.veer) == -1:
            a,b,c = 0.1040, 0.0446, 0.2510
        elif int(self.veer) == 0:
            a,b,c = 0.1043, 0.0445, 0.2510
        elif int(self.veer) == 1:
            a,b,c = 0.1099, 0.0431, 0.2511
        elif int(self.veer) == 2:
            a,b,c = 0.1205, 0.0403, 0.2513
        elif int(self.veer) == 4:
            a,b,c = 0.1535, 0.0316, 0.2518
        elif int(self.veer) == 8:
            a,b,c = 0.2274, 0.0122, 0.2529

        return a * Ct**3 + b * Ct**2 + c * Ct

class GP_Rotor(MomentumModel):
    def __init__(self, veer):
        self.veer  = np.array(veer)
        self._func = self._func_rotor
        self.shear = np.array(0.0)

    def compute_induction(self, aero_props, geom) -> ArrayLike:

        Ct = aero_props.C_x

        # print(Ct)

        Ct = geom.rotor_average(geom.annulus_average(Ct))

        # print(Ct)

        # Ct = np.array(0.69)
        
        GPR = joblib.load('/home1/09909/smata/dir_scratch/induction_modeling/gaussian_process/10MW/results/rotor/opr_kernel.pkl')

        with open('/home1/09909/smata/dir_scratch/induction_modeling/gaussian_process/10MW/train_data/scaler_wrf_cot_rot.pkl', 'rb') as f:
            cot_scalar = pickle.load(f)

        with open('/home1/09909/smata/dir_scratch/induction_modeling/gaussian_process/10MW/train_data/scaler_shears_rot.pkl', 'rb') as f:
            shear_scalar = pickle.load(f)

        with open('/home1/09909/smata/dir_scratch/induction_modeling/gaussian_process/10MW/train_data/scaler_veers_rot.pkl', 'rb') as f:
            veer_scalar = pickle.load(f)

        with open('/home1/09909/smata/dir_scratch/induction_modeling/gaussian_process/10MW//train_data/scaler_wrf_ind_rot.pkl', 'rb') as f:
            ind_scalar = pickle.load(f)

        with open('/home1/09909/smata/dir_scratch/induction_modeling/gaussian_process/10MW//train_data/encoder_rot.pkl', 'rb') as f:
            encoder_rot = pickle.load(f)

        cot_trans   = cot_scalar.transform(Ct.reshape(-1, 1)).ravel()
        shear_trans = shear_scalar.transform(self.shear.reshape(-1, 1)).ravel()
        veer_trans  = veer_scalar.transform(self.veer.reshape(-1, 1)).ravel()

        X_input = np.column_stack([cot_trans, shear_trans, veer_trans])
        X_input = np.hstack([X_input, encoder_rot.transform(np.array([1]).reshape(-1, 1)) ])
        A_pred, std = GPR.predict(X_input, return_std=True)

        # A_pred = np.clip(A_pred, ind_scalar.data_min_, ind_scalar.data_max_)

        a = ind_scalar.inverse_transform(A_pred.reshape(-1, 1)).ravel()

        # print(f'CT:{Ct}')
        # print(f'std:{std}')
        # print(f'sheer:{shear_trans}')
        # print(f'veer:{veer_trans}')

        # print(f'CT:{cot_trans}')
        # print(f'sheer:{shear_trans}')
        # print(f'veer:{veer_trans}')
        # print(f'a:{a}')

        # log_path = "/home1/09909/smata/dir_scratch/induction_modeling/gaussian_process/10MW/results/log_gp_induction.csv"
        # write_header = not os.path.exists(log_path)

        # with open(log_path, "a") as log_file:
        #     if write_header:
        #         log_file.write("Ct,a\n")

        #     # Flatten just in case they are (1,) shaped arrays
        #     Ct_flat = np.ravel(Ct)
        #     a_flat = np.ravel(a)

        #     if Ct_flat.size == 1:
        #         log_file.write(f"{Ct_flat[0]:.6f}, {a_flat[0]:.6f}\n")
        #     else:
        #         for ct_val, a_val in zip(Ct_flat, a_flat):
        #             log_file.write(f"{ct_val:.6f}, {a_val:.6f}\n")

        return a * np.ones_like(geom.mu_mesh)

class GP_Annulus(MomentumModel):
    def __init__(self, veer):
        self.veer  = veer
        self._func = self._func_annulus
        self.shear = 0.0

    def compute_induction(self, aero_props, geom) -> ArrayLike:

        Ct = aero_props.C_x

        Ct = geom.annulus_average(Ct)

        # print(Ct)

        # GPR = joblib.load('/home1/09909/smata/dir_scratch/induction_modeling/gaussian_process/10MW/results/annulus/wrf_10MW_ann_GPR_bk.pkl')
        GPR = joblib.load('/home1/09909/smata/dir_scratch/induction_modeling/gaussian_process/10MW/results/annulus/opr_kernel.pkl')

        with open('/home1/09909/smata/dir_scratch/induction_modeling/gaussian_process/10MW/train_data/scaler_wrf_cot_ann.pkl', 'rb') as f:
            cot_scalar = pickle.load(f)

        with open('/home1/09909/smata/dir_scratch/induction_modeling/gaussian_process/10MW/train_data/scaler_shears_ann.pkl', 'rb') as f:
            shear_scalar = pickle.load(f)

        with open('/home1/09909/smata/dir_scratch/induction_modeling/gaussian_process/10MW/train_data/scaler_veers_ann.pkl', 'rb') as f:
            veer_scalar = pickle.load(f)

        with open('/home1/09909/smata/dir_scratch/induction_modeling/gaussian_process/10MW//train_data/scaler_wrf_ind_ann.pkl', 'rb') as f:
            ind_scalar = pickle.load(f)

        with open('/home1/09909/smata/dir_scratch/induction_modeling/gaussian_process/10MW//train_data/encoder_ann.pkl', 'rb') as f:
            encoder_ann = pickle.load(f)

        cot_trans = cot_scalar.transform(Ct.reshape(-1, 1)).ravel()
        shear_val = shear_scalar.transform(np.array(self.shear).reshape(-1, 1)).item()
        veer_val  = veer_scalar.transform(np.array(self.veer).reshape(-1, 1)).item()

        # Broadcast to shape of Ct
        shear_trans = np.full_like(Ct, shear_val)
        veer_trans  = np.full_like(Ct, veer_val)

        X_input = np.column_stack([geom.mu, cot_trans, shear_trans, veer_trans])

        one_hot = encoder_ann.transform(np.ones_like(geom.mu).reshape(-1, 1))

        X_input = np.hstack([X_input, one_hot])

        A_pred, std = GPR.predict(X_input, return_std=True)

        a = ind_scalar.inverse_transform(A_pred.reshape(-1, 1)).ravel()

        save_dir = '/scratch/09909/smata/temp_4'
        os.makedirs(save_dir, exist_ok=True)

        idx = len(os.listdir(save_dir)) // 2  # assumes one Ct + one a per call
        np.save(os.path.join(save_dir, f'Ct_{idx:04d}.npy'), Ct)
        np.save(os.path.join(save_dir, f'a_{idx:04d}.npy'), a)

        if self.veer == 0:
            a = np.load('/scratch/09909/smata/induction_modeling/gaussian_process/10MW/train_data/wrf_ind_ann.npy')

            a = a[:,31]

            return a[:, np.newaxis] * np.ones_like(geom.mu_mesh)

        # print(f'a:{a}')

        # return np.tile(a[:, np.newaxis], (1, geom.Ntheta))
        return a[:, np.newaxis] * np.ones_like(geom.mu_mesh)