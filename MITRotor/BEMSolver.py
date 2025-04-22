from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from UnifiedMomentumModel.Momentum import Heck
from UnifiedMomentumModel.Utilities.FixedPointIteration import FixedPointIterationResult, adaptivefixedpointiteration

from . import Momentum, TipLoss
from .Aerodynamics import AerodynamicModel, AerodynamicProperties, DefaultAerodynamics
from .Geometry import BEMGeometry
from .RotorDefinition import RotorDefinition
from .TangentialInduction import DefaultTangentialInduction, TangentialInductionModel


def average(geometry: BEMGeometry, value: ArrayLike, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
    # Assuming the function returns a 2D grid of values

    if grid == "sector":
        # No averaging
        return value

    elif grid == "annulus":
        # Average over azimuthal sectors.
        return geometry.annulus_average(value)

    elif grid == "rotor":
        # Average of entire rotor
        return geometry.rotor_average(geometry.annulus_average(value))

    else:
        raise ValueError(f"Unsupported grid averaging type: {grid}")


@dataclass
class BEMSolution:
    pitch: float
    tsr: float
    yaw: float
    v_inf: float
    aero_props: AerodynamicProperties = field(repr=False)
    geom: BEMGeometry = field(repr=False)
    rotor: RotorDefinition
    converged: bool
    niter: int

    def __post_init__(self):
        sol = Heck()(self.Ctprime(), self.yaw)
        self._u4, self._v4 = sol.u4, sol.v4

    def a(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        return average(self.geom, self.aero_props.an, grid)

    def aprime(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        return average(self.geom, self.aero_props.aprime, grid)

    def solidity(self, grid: Literal["sector ", "annulus", "rotor"] = "rotor"):
        return average(self.geom, self.aero_props.solidity, grid)

    def U(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        return average(self.geom, self.aero_props.U, grid)

    def wdir(self, grid: Literal["sector ", "annulus", "rotor"] = "rotor"):
        return average(self.geom, self.aero_props.wdir, grid)

    def Vax(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        return average(self.geom, self.aero_props.Vax, grid)

    def Vtan(self, grid: Literal["sector ", "annulus", "rotor"] = "rotor"):
        return average(self.geom, self.aero_props.Vtan, grid)

    def W(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        return average(self.geom, self.aero_props.W, grid)

    def phi(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        return average(self.geom, self.aero_props.phi, grid)

    def aoa(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        return average(self.geom, self.aero_props.aoa, grid)

    def Cl(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        return average(self.geom, self.aero_props.Cl, grid)

    def Cd(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        return average(self.geom, self.aero_props.Cd, grid)
    
    def Cn(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        return average(self.geom, self.aero_props.C_n, grid)
    
    def Ctan(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        return average(self.geom, self.aero_props.C_tan, grid)

    def Cx(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        return average(self.geom, self.aero_props.C_x_corr, grid)

    def Ctau(self, grid: Literal["sector ", "annulus", "rotor"] = "rotor"):
        return average(self.geom, self.aero_props.C_tau_corr, grid)
    
    def Ctau_uncorr(self, grid: Literal["sector ", "annulus", "rotor"] = "rotor"):
        return average(self.geom, self.aero_props.C_tau, grid)

    def F(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        return average(self.geom, self.aero_props.F, grid)

    def u4(self):
        return self._u4

    def v4(self):
        return self._v4

    def Cp(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        dCp = (
            self.tsr
            * self.geom.mu_mesh
            * self.Ctau_uncorr(grid="sector")
        )
        return average(self.geom, dCp, grid=grid)
    
    def Cp_corr(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        dCp = (
            self.tsr
            * self.geom.mu_mesh
            * self.Ctau(grid="sector")
        )
        return average(self.geom, dCp, grid=grid)

    def power(self):

        if not self.v_inf == 1.0:
            rho = 1.225

            # Compute rotational speed (ω)
            omega = (self.tsr *self.v_inf) / self.rotor.R

            # Get normalized radial positions and azimuthal angles
            theta = self.geom.theta_mesh
            r_dim = self.geom.mu_mesh * self.rotor.R

            # Dimensionalize velocity
            W_dim = self.W(grid="sector") * self.v_inf

            # Compute solidity
            solidity = self.solidity(grid="sector")

            # Compute differential torque (dQ) using solidity
            dQ = (np.pi * rho * solidity * W_dim**2 * self.Ctan(grid="sector") * r_dim**2)

            # Integrate over r 
            Q_r_integrated = np.trapz(dQ, x=r_dim, axis=0)

            # Integrate over theta 
            Q_total = np.trapz(Q_r_integrated, x=theta[0, :], axis=0)

            # Compute total power
            P_total = (omega / (2 * np.pi)) * Q_total

        else:
            P_total = np.nan

        return P_total

    def thrust(self):

        if not self.v_inf == 1.0:
            rho = 1.225

            # Get normalized radial positions and azimuthal angles
            theta = self.geom.theta_mesh
            r_dim = self.geom.mu_mesh * self.rotor.R 

            # Dimensionalize velocity
            W_dim = self.W(grid="sector") * self.v_inf

            # Compute solidity
            solidity = self.solidity(grid="sector")

            # Compute differential torque (dQ) using solidity
            dT = (np.pi * rho * solidity * W_dim**2 * self.Cax(grid="sector") * r_dim)

            # Integrate over r 
            T_r_integrated = np.trapz(dT, x=r_dim, axis=0)

            # Integrate over theta
            T_total = np.trapz(T_r_integrated, x=theta[0, :], axis=0)

            # Compute total power
            T_total = (1 / (2 * np.pi)) * T_total

        else:
            T_total = np.nan

        return T_total

    def Ct(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        _Ct = self.aero_props.C_x
        return average(self.geom, _Ct, grid=grid)
    
    def Ct_corr(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        _Ct = self.aero_props.C_x_corr
        return average(self.geom, _Ct, grid=grid)

    def Ctprime(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        Ctprime = self.Ct(grid="sector") / ((1 - self.a(grid="sector")) ** 2 * np.cos(self.yaw) ** 2)
        return average(self.geom, Ctprime, grid=grid)


@adaptivefixedpointiteration(max_iter=500, relaxations=[0.25, 0.5, 0.96])
class BEM:
    """
    A generic BEM class which facilitates dependency injection for various models.
    Models which can be injected are:
    - rotor definition
    - BEM geometry
    - aerodynamic properties calculation method
    - tip loss method
    - axial induction calculation method
    - tangential induction calculation method
    """

    def __init__(
        self,
        rotor: RotorDefinition,
        geometry: Optional[BEMGeometry] = None,
        tiploss_model: Optional[TipLoss.TipLossModel] = None,
        momentum_model: Optional[Momentum.MomentumModel] = None,
        tangential_induction_model: Optional[TangentialInductionModel] = None,
        aerodynamic_model: Optional[AerodynamicModel] = None,
    ):
        self.rotor = rotor

        self.geometry: BEMGeometry = geometry or BEMGeometry(Nr=10, Ntheta=20)
        self.aerodynamic_model = aerodynamic_model or DefaultAerodynamics()
        self.tiploss_model: TipLoss.TipLossModel = tiploss_model or TipLoss.PrandtlTipLoss(root_loss=True)
        self.momentum_model: Momentum.MomentumModel = momentum_model or Momentum.HeckMomentum()
        self.tangential_induction_model = tangential_induction_model or DefaultTangentialInduction()

        # self._solidity = self.rotor.solidity(self.geometry.mu)

    def __call__(self, pitch: float, tsr: float, yaw: float, v_inf: float = 1.0, a: float = 1/3) -> BEMSolution:
        ...

    def sample_points(self, yaw: float = 0.0) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        X, Y, Z = self.geometry.cartesian(yaw)
        return X, Y, Z

    def initial_guess(
        self, pitch: float, tsr: float, yaw: float = 0.0, U: ArrayLike = 1.0, wdir: ArrayLike = 0.0
    ) -> Tuple[ArrayLike, ...]:
        a = (1 / 3) * np.ones(self.geometry.shape)
        aprime = np.zeros(self.geometry.shape)

        return a, aprime

    def residual(
        self,
        x: Tuple[ArrayLike, ...],
        pitch: ArrayLike,
        tsr: ArrayLike,
        yaw: ArrayLike = 0.0,
        v_inf: ArrayLike = 1.0,
        U: ArrayLike = 1.0,
        wdir: ArrayLike = 0.0,
        a: float = 1/2,
    ) -> Tuple[ArrayLike, ...]:
        an, aprime = x

        aero_props = self.aerodynamic_model(
            an = an, 
            aprime=aprime, 
            pitch=pitch, 
            tsr=tsr, 
            yaw=yaw, 
            rotor=self.rotor, 
            geom=self.geometry, 
            U=U, 
            wdir=wdir)
        aero_props.F = self.tiploss_model(aero_props, pitch, tsr, yaw, self.rotor, self.geometry)
        e_an = self.momentum_model(aero_props, pitch, tsr, yaw, self.rotor, self.geometry, a=a) - an
        e_aprime = self.tangential_induction_model(aero_props, pitch, tsr, yaw, self.rotor, self.geometry) - aprime

        return e_an, e_aprime

    def post_process(self, result: FixedPointIterationResult, pitch, tsr, yaw, v_inf=1.0, U=1.0, wdir=0.0) -> BEMSolution:
        U = np.ones(self.geometry.shape) if U is None else U
        wdir = np.zeros(self.geometry.shape) if wdir is None else wdir
        an, aprime = result.x
        aero_props = self.aerodynamic_model(an, aprime, pitch, tsr, yaw, self.rotor, self.geometry, U, wdir)
        aero_props.F = self.tiploss_model(aero_props, pitch, tsr, yaw, self.rotor, self.geometry)

        return BEMSolution(pitch, tsr, yaw, v_inf, aero_props, self.geometry, self.rotor, result.converged, result.niter)
