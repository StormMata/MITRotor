from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from UnifiedMomentumModel.Utilities.FixedPointIteration import FixedPointIterationResult, adaptivefixedpointiteration

from . import Momentum, TipLoss
from .Aerodynamics import AerodynamicModel, AerodynamicProperties, DefaultAerodynamics
from .Geometry import BEMGeometry
from .RotorDefinition import RotorDefinition
from .TangentialInduction import DefaultTangentialInduction, TangentialInductionModel
from UnifiedMomentumModel.Utilities.Geometry import calc_eff_yaw

from scipy.optimize import brentq


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
    rotor: RotorDefinition = field(repr=False)
    aero_props: AerodynamicProperties = field(repr=False)
    geom: BEMGeometry = field(repr=False)
    converged: bool
    niter: int
    u4: float
    v4: float
    tilt: float = 0.0
    w4: float = 0
    U_ref: Optional[float] = None
    rho: float = 1.225

    def a(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        return average(self.geom, self.aero_props.an, grid)

    def aprime(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        return average(self.geom, self.aero_props.aprime, grid)

    def solidity(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        return average(self.geom, self.aero_props.solidity, grid)

    def U(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        return average(self.geom, self.aero_props.U, grid)

    def wdir(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
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

    def Ctau(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        return average(self.geom, self.aero_props.C_tau_corr, grid)
    
    def Ctau_uncorr(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        return average(self.geom, self.aero_props.C_tau, grid)

    def F(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        return average(self.geom, self.aero_props.F, grid)
    
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

    def Ct(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        _Ct = self.aero_props.C_x
        return average(self.geom, _Ct, grid=grid)
    
    def Ct_corr(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        _Ct = self.aero_props.C_x_corr
        return average(self.geom, _Ct, grid=grid)

    def Ctprime(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        eff_yaw = calc_eff_yaw(self.yaw, self.tilt)
        Ctprime = self.Ct(grid="sector") / ((1 - self.a(grid="sector")) ** 2 * np.cos(eff_yaw) ** 2)
        return average(self.geom, Ctprime, grid=grid)
    
    def Cq(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        dCq = (
            self.geom.mu_mesh
            * self.Ctau_uncorr(grid="sector")
        )
        return average(self.geom, dCq, grid=grid)


    def Cq_corr(self, grid: Literal["sector", "annulus", "rotor"] = "rotor"):
        dCq = (
            self.geom.mu_mesh
            * self.Ctau(grid="sector")
        )
        return average(self.geom, dCq, grid=grid)

    # def thrust(self):

    #     dT = 1/2 * 3 * self.rho * self.rotor.chord_func(self.geom.mu_mesh) * (self.W('sector') * self.U_ref)**2 * (self.Cl('sector') * np.cos(self.phi('sector')) + self.Cd('sector') * np.sin(self.phi('sector')))

    #     Thrust = np.trapezoid(np.trapezoid(dT, self.geom.theta_mesh, axis=1), self.geom.mu * self.rotor.R) / (2 * np.pi)

    #     return Thrust
    
    # def torque(self):

    #     dQ = 1/2 * 3 * self.rho * self.rotor.chord_func(self.geom.mu_mesh) * (self.W('sector') * self.U_ref)**2 * (self.Cl('sector') * np.sin(self.phi('sector')) - self.Cd('sector') * np.cos(self.phi('sector'))) * self.geom.mu_mesh * self.rotor.R

    #     Torque = np.trapezoid(np.trapezoid(dQ, self.geom.theta_mesh, axis=1), self.geom.mu * self.rotor.R) / (2 * np.pi)

    #     return Torque
    
    # def power(self):

    #     dQ = 1/2 * 3 * self.rho * self.rotor.chord_func(self.geom.mu_mesh) * (self.W('sector') * self.U_ref)**2 * (self.Cl('sector') * np.sin(self.phi('sector')) - self.Cd('sector') * np.cos(self.phi('sector'))) * self.geom.mu_mesh * self.rotor.R

    #     # print(self.rotor.chord_func(self.geom.mu_mesh))

    #     dP = dQ * self.tsr * self.U_ref / self.rotor.R

    #     Power = np.trapezoid(np.trapezoid(dP, self.geom.theta_mesh, axis=1), self.geom.mu * self.rotor.R) / (2 * np.pi)

    #     return Power
    
    def thrust(self):
        B = self.rotor.N_blades
        c = self.rotor.chord_func(self.geom.mu_mesh)
        Wdim = self.W("sector") * self.U_ref

        Cn = (
            self.Cl("sector") * np.cos(self.phi("sector"))
            + self.Cd("sector") * np.sin(self.phi("sector"))
        )

        dT = 0.5 * B * self.rho * c * Wdim**2 * Cn  # N/m

        dr = self.rotor.R * np.diff(self.geom.mu_edges)  # shape (Nr,)

        T_annulus_mean = np.mean(dT, axis=1)  # average over theta

        return np.sum(T_annulus_mean * dr)

    def torque(self):
        B = self.rotor.N_blades
        c = self.rotor.chord_func(self.geom.mu_mesh)
        r = self.geom.mu_mesh * self.rotor.R
        Wdim = self.W("sector") * self.U_ref

        Ctangential = (
            self.Cl("sector") * np.sin(self.phi("sector"))
            - self.Cd("sector") * np.cos(self.phi("sector"))
        )

        dQ = 0.5 * B * self.rho * c * Wdim**2 * Ctangential * r  # N m / m

        dr = self.rotor.R * np.diff(self.geom.mu_edges)

        Q_annulus_mean = np.mean(dQ, axis=1)

        return np.sum(Q_annulus_mean * dr)

    def power(self):
        omega = self.tsr * self.U_ref / self.rotor.R
        return omega * self.torque()
            
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
        self.tangential_induction_model = tangential_induction_model or DefaultTangentialInduction()
        # need to pass in a momentum model from MITRotor - NOT from UMM
        if momentum_model is not None and not isinstance(momentum_model, Momentum.MomentumModel):
            raise TypeError(f"Expected MomentumModel from MITRotor or None, got {type(momentum_model).__name__}")
        self.momentum_model: Momentum.MomentumModel = momentum_model or Momentum.HeckMomentum()

        # self._solidity = self.rotor.solidity(self.geometry.mu)

    def sample_points(self, yaw: float = 0.0, tilt: float = 0.0) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        X, Y, Z = self.geometry.cartesian(yaw, tilt)
        return X, Y, Z

    def initial_guess(self, *args, **kwargs) -> Tuple[ArrayLike, ...]:
        a = (1 / 3) * np.ones(self.geometry.shape)
        aprime = np.zeros(self.geometry.shape)

        return a, aprime

    def residual(
        self,
        x: Tuple[ArrayLike, ...],
        pitch: ArrayLike,
        tsr: ArrayLike,
        yaw: ArrayLike = 0.0,
        U: ArrayLike = None,
        wdir: ArrayLike = None,
        tilt: ArrayLike = 0.0,
        U_ref: float = 0.0,
        rho: float = 1.225,
    ) -> Tuple[ArrayLike, ...]:
        an, aprime = x
        U = np.ones(self.geometry.shape) if U is None else U
        wdir = np.zeros(self.geometry.shape) if wdir is None else wdir

        aero_props = self.aerodynamic_model(
            an = an, 
            aprime=aprime, 
            pitch=pitch, 
            tsr=tsr, 
            yaw=yaw, 
            rotor=self.rotor, 
            geom=self.geometry, 
            U=U, 
            wdir=wdir,
            tilt = tilt,
        )
    
        aero_props.F = self.tiploss_model(aero_props, pitch, tsr, yaw, self.rotor, self.geometry, tilt = tilt)
        e_an = self.momentum_model(aero_props, pitch, tsr, yaw, self.rotor, self.geometry, tilt = tilt) - an
        e_aprime = self.tangential_induction_model(aero_props, pitch, tsr, yaw, self.rotor, self.geometry, tilt = tilt) - aprime

        return e_an, e_aprime


    def post_process(
        self,
        result: FixedPointIterationResult,
        pitch,
        tsr,
        yaw=0,
        U_ref=1.0,
        U=None,
        wdir=None,
        tilt=0.0,
        rho: float = 1.225,
    ) -> BEMSolution:
        U = np.ones(self.geometry.shape) if U is None else U
        wdir = np.zeros(self.geometry.shape) if wdir is None else wdir
        an, aprime = result.x
        aero_props = self.aerodynamic_model(an, aprime, pitch, tsr, yaw, self.rotor, self.geometry, U, wdir, tilt = tilt)
        aero_props.F = self.tiploss_model(aero_props, pitch, tsr, yaw, self.rotor, self.geometry, tilt = tilt)
        avg_Ct = average(self.geometry, aero_props.C_x)
        u4,v4,w4 = self.momentum_model.compute_initial_wake_velocities(avg_Ct, yaw, tilt = tilt)

        return BEMSolution(
            pitch=pitch,
            tsr=tsr,
            yaw=yaw,
            rotor=self.rotor,
            aero_props=aero_props,
            geom=self.geometry,
            converged=result.converged,
            niter=result.niter,
            u4=u4,
            v4=v4,
            tilt=tilt,
            w4=w4,
            U_ref=U_ref,
            rho=rho,
        )
    
class BEMWithController:
    def __init__(self, bem: BEM):
        self.bem = bem
        self.rotor = bem.rotor
        self.geometry = bem.geometry

    def __getattr__(self, name):
        return getattr(self.bem, name)

    def __call__(
        self,
        pitch: float,
        U: ArrayLike = None,
        wdir: ArrayLike = None,
        tsr: Optional[float] = None,
        tsr_mode: Literal["given", "komega"] = "given",
        k: Optional[float] = None,
        yaw: float = 0.0,
        tilt: float = 0.0,
        U_ref: Optional[float] = None,
        rho: float = 1.225,
        omega_min: float = 0.5,
        omega_max: float = 2.0,
        xtol: float = 1e-3,
        rtol: float = 1e-3,
    ):
        if tsr_mode == "given":
            if tsr is None:
                raise ValueError("tsr must be provided when tsr_mode='given'.")
            sol = self.bem(
                pitch=pitch,
                tsr=tsr,
                yaw=yaw,
                U=U,
                wdir=wdir,
                tilt=tilt,
                U_ref=U_ref,
                rho=rho,
            )
            sol.omega = None
            sol.tsr_mode = "given"
            return sol

        if tsr_mode != "komega":
            raise ValueError(f"Unsupported tsr_mode: {tsr_mode}")

        if k is None:
            raise ValueError("k must be provided when tsr_mode='komega'.")

        if U_ref is None:
            U_arr = np.asarray(U, dtype=float)
            U_ref = float(np.nanmean(U_arr))

        if U_ref <= 0:
            raise ValueError("U_ref must be positive.")

        R = self.rotor.R

        def residual_omega(omega: float) -> float:
            tsr_trial = omega * R / U_ref

            sol_trial = self.bem(
                pitch=pitch,
                tsr=tsr_trial,
                yaw=yaw,
                U=U,
                wdir=wdir,
                tilt=tilt,
                U_ref=U_ref,
                rho=rho,
            )

            Q_aero = sol_trial.torque()
            Q_gen = k * omega**2
            return Q_aero - Q_gen

        f_lo = residual_omega(omega_min)
        f_hi = residual_omega(omega_max)

        if np.isnan(f_lo) or np.isnan(f_hi):
            raise RuntimeError("Controller residual is NaN at omega bounds.")

        if f_lo * f_hi > 0:
            raise RuntimeError(
                "Could not bracket controller operating point: "
                f"residual({omega_min})={f_lo:.6g}, "
                f"residual({omega_max})={f_hi:.6g}"
            )

        omega = brentq(residual_omega, omega_min, omega_max, xtol=xtol, rtol=rtol)
        tsr = omega * R / U_ref

        sol = self.bem(
            pitch=pitch,
            tsr=tsr,
            yaw=yaw,
            U=U,
            wdir=wdir,
            tilt=tilt,
            U_ref=U_ref,
            rho=rho,
        )

        sol.omega = float(omega)
        sol.tsr_mode = "komega"
        sol.k = float(k)
        sol.U_ref = float(U_ref)
        return sol