from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from .RotorDefinition import RotorDefinition
from .Geometry import BEMGeometry
from UnifiedMomentumModel.Utilities.Geometry import calc_eff_yaw

__all__ = [
    "AerodynamicModel",
    "AerodynamicProperties",
    "DefaultAerodynamics",
    "KraghAerodynamics",
    "WRFLESAerodynamics",
]


@dataclass
class AerodynamicProperties:
    """
    Data class representing aerodynamic properties.

    Attributes (on radial grid):
        an (ArrayLike): Axial induction
        aprime (ArrayLike): Tangential induction
        solidity (ArrayLike): Blade solidity

    Attributes (on polar grid):
        U (ArrayLike): Inflow velocity.
        wdir (ArrayLike): Inflow direction.
        Vax (ArrayLike): Blade element axial velocity.
        Vtan (ArrayLike): Blade element tangential velocity.
        aoa (ArrayLike): Blade element angle of attack.
        Cl (ArrayLike): Blade element lift coefficient.
        Cd (ArrayLike): Blade element drag coefficient.
        F (ArrayLike): Blade element tip loss.

    Properties:
        W: Blade element inflow magnitude.
        phi: Blade element inflow direction
        Ctan: Blade element tangengial force coefficient.
        Cax: Blade element axial force coefficient.
    """

    # Radial grid
    an: ArrayLike
    aprime: ArrayLike
    solidity: ArrayLike
    # Full grid
    U: ArrayLike
    wdir: ArrayLike
    Vax: ArrayLike
    Vtan: ArrayLike
    aoa: ArrayLike
    Cl: ArrayLike
    Cd: ArrayLike
    F: ArrayLike = None

    def __post_init__(self):
        pass

    @cached_property
    def W(self):
        """
        Blade element inflow magnitude.
        """
        return np.sqrt(self.Vax**2 + self.Vtan**2)

    @cached_property
    def phi(self):
        """
        Blade element inflow direction.
        """
        return np.arctan2(self.Vax, self.Vtan)
    
    @cached_property
    def C_n(self):
        """
        Blade element axial blade force coefficient.
        """
        return self.Cl * np.cos(self.phi) + self.Cd * np.sin(self.phi)

    @cached_property
    def C_tan(self):
        """
        Blade element tangential blade force coefficient.
        """
        return self.Cl * np.sin(self.phi) - self.Cd * np.cos(self.phi)
    
    @cached_property
    def C_x(self):
        """
        Blade element axial area force coefficient.
        """
        return self.solidity * self.W**2 * self.C_n

    @cached_property
    def C_tau(self):
        """
        Blade element tangential area force coefficient.
        """
        return self.solidity * self.W**2 * self.C_tan
    
    @cached_property
    def C_x_corr(self):
        """
        Corrected blade element area axial force coefficient.
        """
        return self.C_x / self.F
    
    @cached_property
    def C_tau_corr(self):
        """
        Corrected blade element area tangential force coefficient.
        """
        return self.C_tau / self.F



class AerodynamicModel(ABC):
    @abstractmethod
    def __call__(
        self,
        an: ArrayLike,
        aprime: ArrayLike,
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: RotorDefinition,
        geom: BEMGeometry,
        U: ArrayLike,
        wdir: ArrayLike,
        tilt: float = 0,
    ) -> AerodynamicProperties:
        """
        Performs the aerodynamic calculations in a blade-element code.

        Args:
            an (ArrayLike): Axial induction radial profile.
            aprime (ArrayLike): tangengial induction radial profile.
            pitch (float): blade pitch angle [rad].
            tsr (float): Rotor tip-speed ratio.
            yaw (float): Rotor yaw angle [rad].
            rotor (RotorDefinition): Turbine rotor definition object.
            geom (BEMGeometry): Blade element geometry object.
            U (ArrayLike): Inflow velocity on polar grid.
            wdir (ArrayLike): Inflow direction on polar grid.
            tilt (float): Rotor tilt angle [rad].

        Returns:
            AerodynamicProperties: Calculated aerodynamic properties stored in AerodynamicProperties object.

        """
        ...


class KraghAerodynamics(AerodynamicModel):
    def __call__(
        self,
        an: ArrayLike,
        aprime: ArrayLike,
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: RotorDefinition,
        geom: BEMGeometry,
        U: ArrayLike,
        wdir: ArrayLike,
        tilt: float = 0.0,
    ) -> AerodynamicProperties:
        """
        Performs the aerodynamic calculations in a blade-element code using the
        method outlined in Howland et al. 2020. (Influence of atmospheric conditions
        on the power production of utility-scale wind turbines in yaw misalignment),
        which builds on 2014 paper by Kragh and Hansen: https://doi.org/10.1002/we.1612.

        Args:
            an (ArrayLike): Axial induction radial profile.
            aprime (ArrayLike): tangengial induction radial profile.
            pitch (float): blade pitch angle [rad].
            tsr (float): Rotor tip-speed ratio.
            yaw (float): Rotor yaw angle [rad].
            rotor (RotorDefinition): Turbine rotor definition object.
            geom (BEMGeometry): Blade element geometry object.
            U (ArrayLike): Inflow velocity on polar grid.
            wdir (ArrayLike): Inflow direction on polar grid.
            tilt (float): Rotor tilt angle [rad].

        Returns:
            AerodynamicProperties: Calculated aerodynamic properties stored in AerodynamicProperties object.

        """
        if tilt != 0:
            raise ValueError("Tilt not supported by the KraghAerodynamics model. Use DefaultAerodynamics.")
        local_yaw = wdir - yaw

        Vax = (
            U
            * (1 - an)
            * np.cos(local_yaw * np.cos(geom.theta_mesh))
            * np.cos(local_yaw * np.sin(geom.theta_mesh))
        )
        Vtan = (
            (1 + aprime) * tsr * geom.mu_mesh
            - U * (1 - an)
            * np.cos(local_yaw * np.sin(geom.theta_mesh))
            * np.sin(local_yaw * np.cos(geom.theta_mesh))
        )

        phi = np.arctan2(Vax, Vtan)
        aoa = phi - rotor.twist(geom.mu_mesh) - pitch
        aoa = np.clip(aoa, -np.pi / 2, np.pi / 2)

        Cl, Cd = rotor.clcd(geom.mu_mesh, aoa)

        solidity = rotor.solidity(geom.mu_mesh)

        aero_props = AerodynamicProperties(
            an = an,
            aprime = aprime,
            solidity = solidity,
            U = U * np.ones(geom.shape),
            wdir = wdir * np.ones(geom.shape),
            Vax = Vax,
            Vtan = Vtan,
            aoa = aoa,
            Cl = Cl,
            Cd = Cd,
        )

        return aero_props


class DefaultAerodynamics(AerodynamicModel):
    def __call__(
        self,
        an: ArrayLike,
        aprime: ArrayLike,
        pitch: float,
        tsr: float,
        yaw: float,
        rotor: RotorDefinition,
        geom: BEMGeometry,
        U: ArrayLike,
        wdir: ArrayLike,
        tilt: float = 0.0,
    ) -> AerodynamicProperties:
        """
        Performs the aerodynamic calculations in a blade-element code using the
        method outlined in the supplementary material in Liew et al., 2024:
        https://www.nature.com/articles/s41467-024-50756-5

        Args:
            an (ArrayLike): Axial induction radial profile.
            aprime (ArrayLike): tangengial induction radial profile.
            pitch (float): blade pitch angle [rad].
            tsr (float): Rotor tip-speed ratio.
            yaw (float): Rotor yaw angle [rad].
            rotor (RotorDefinition): Turbine rotor definition object.
            geom (BEMGeometry): Blade element geometry object.
            U (ArrayLike): Inflow velocity on polar grid.
            wdir (ArrayLike): Inflow direction on polar grid.

        Returns:
            AerodynamicProperties: Calculated aerodynamic properties stored in AerodynamicProperties object.

        """
        # calculate values in "yaw-only" frame
        local_yaw = -self.eff_yaw
        Vax = U * ((1 - an) * np.cos(local_yaw))
        Vtan = (
            (1 + aprime) * tsr * geom.mu_mesh
            - U * (1 - an)
            * np.cos(self.eff_theta_mesh)
            * np.sin(local_yaw)
        )

        phi = np.arctan2(Vax, Vtan)
        aoa = phi - rotor.twist(geom.mu_mesh) - pitch
        aoa = np.clip(aoa, -np.pi / 2, np.pi / 2)

        Cl, Cd = rotor.clcd(geom.mu_mesh, aoa)

        solidity = rotor.solidity(geom.mu_mesh)

        aero_props = AerodynamicProperties(
            an = an,
            aprime = aprime,
            solidity = solidity,
            U = U * np.ones(geom.shape),
            wdir = wdir * np.ones(geom.shape),
            Vax = Vax,
            Vtan = Vtan,
            aoa = aoa,
            Cl = Cl,
            Cd = Cd,
        )

        return aero_props


# class WRFLESAerodynamics(AerodynamicModel):
#     def __call__(
#         self,
#         an: ArrayLike,
#         aprime: ArrayLike,
#         pitch: float,
#         tsr: float,
#         yaw: float,
#         rotor: RotorDefinition,
#         geom: BEMGeometry,
#         U: ArrayLike,
#         wdir: ArrayLike,
#         tilt: float = 0.0,
#         precone: float = 0.0,
#     ) -> AerodynamicProperties:
#         """
#         Performs the aerodynamic calculations in a blade-element code using the
#         equations in WRF-LES as implemented by Kale et al. (2022) (see eq. B.3):
#         https://doi.org/10.1016/j.renene.2022.07.119

#         Equations are simplified assuming no cone and no tilt. Vertical velocity
#         term in the equation for Vtan is neglected.

#         Args:
#             an (ArrayLike): Axial induction radial profile.
#             aprime (ArrayLike): tangengial induction radial profile.
#             pitch (float): blade pitch angle [rad].
#             tsr (float): Rotor tip-speed ratio.
#             yaw (float): Rotor yaw angle [rad].
#             rotor (RotorDefinition): Turbine rotor definition object.
#             geom (BEMGeometry): Blade element geometry object.
#             U (ArrayLike): Inflow velocity on polar grid.
#             wdir (ArrayLike): Inflow direction on polar grid.

#         Returns:
#             AerodynamicProperties: Calculated aerodynamic properties stored in AerodynamicProperties object.

#         """

#         u_fst = (U * (1 - an) * np.cos(wdir))
#         v_fst = (U * (1 - an) * np.sin(wdir))

#         # u_inf = U * np.cos(wdir)
#         # v_inf = U * np.sin(wdir)

#         # u_fst = ((((1-an) * u_inf)**2 + v_inf**2)**(1/2) * np.cos(np.atan2(v_inf, u_inf)))
#         # v_fst = ((((1-an) * u_inf)**2 + v_inf**2)**(1/2) * np.sin(np.atan2(v_inf, u_inf)))

#         w_fst = np.zeros_like(u_fst)

#         Vax, Vtn_NR, _ = WRFLESAerodynamics.rotGlobalToLocal(geom.Nr,geom.Ntheta,u_fst,v_fst,w_fst, yaw, tilt, precone)

#         Vtan = (1 + aprime) * tsr * geom.mu_mesh - Vtn_NR

#         phi = np.arctan2(Vax, Vtan)
#         aoa = phi - rotor.twist(geom.mu_mesh) - pitch
#         aoa = np.clip(aoa, -np.pi / 2, np.pi / 2)

#         Cl, Cd = rotor.clcd(geom.mu_mesh, aoa)

#         solidity = rotor.solidity(geom.mu_mesh)

#         aero_props = AerodynamicProperties(
#             an = an,
#             aprime = aprime,
#             solidity = solidity,
#             U = U * np.ones(geom.shape),
#             wdir = wdir * np.ones(geom.shape),
#             Vax = Vax,
#             Vtan = Vtan,
#             aoa = aoa,
#             Cl = Cl,
#             Cd = Cd,
#         )

#         return aero_props

#     @staticmethod
#     def rotGlobalToLocal(Nelm,Nsct,u_rotor,v_rotor,w_rotor, yaw, tilt, precone):
#         """
#         Replicates the matrix equations implemented in WRF-LES

#         Args:


#         Returns:
#             Axial, tangential (wihtout rotation), and radial velocity components pointwise over the rotor

#         """
#         precone = precone
#         tilt    = tilt
#         trbYaw  = yaw

#         psi = 0.0
#         angle = 2 * np.pi / Nsct

#         Ux   = np.zeros_like(u_rotor,dtype='float')
#         Utau = np.zeros_like(u_rotor,dtype='float')
#         Ur   = np.zeros_like(u_rotor,dtype='float')

#         for i in range(Nelm):
#             for j in range(Nsct):
#                 transposePrecone = np.array([[ np.cos(precone), 0,  np.sin(precone)],
#                                             [0,                1,                0],
#                                             [-np.sin(precone), 0,  np.cos(precone)]])

#                 transposeAzimuth = np.array([[1,                0,                0],
#                                             [0,      np.cos(psi),      np.sin(psi)],
#                                             [0,     -np.sin(psi),      np.cos(psi)]])

#                 transposeTilt    = np.array([[np.cos(tilt),     0,    -np.sin(tilt)],
#                                             [0,                1,                0],
#                                             [np.sin(tilt),     0,     np.cos(tilt)]])

#                 transposeYaw     = np.array([[np.cos(trbYaw),  np.sin(trbYaw),    0],
#                                             [-np.sin(trbYaw), np.cos(trbYaw),    0],
#                                             [0,                0,                1]])

#                 psi = psi + angle

#                 PreconeAzimuth        = np.matmul(transposePrecone,   transposeAzimuth)
#                 PreconeAzimuthTilt    = np.matmul(PreconeAzimuth,     transposeTilt)
#                 PreconeAzimuthTiltYaw = np.matmul(PreconeAzimuthTilt, transposeYaw)

#                 local = np.matmul(PreconeAzimuthTiltYaw, np.array([[u_rotor[i,j]], [v_rotor[i,j]], [w_rotor[i,j]]]))

#                 Ux[i,j]   = local[0][0]
#                 Utau[i,j] = local[1][0]
#                 Ur[i,j]   = local[2][0]

#         return Ux, Utau, Ur
    

class WRFLESAerodynamics(AerodynamicModel):
    def __call__(
        self,
        an,
        aprime,
        pitch,
        tsr,
        yaw,
        rotor,
        geom,
        U,
        wdir,
        tilt=0.0,
        precone=0.0,
    ):
        if tilt != 0.0 or precone != 0.0:
            raise ValueError("Fast path only supports tilt=0 and precone=0.")

        # Induced inflow in the fixed frame
        u_fst = U * (1 - an) * np.cos(wdir)
        v_fst = U * (1 - an) * np.sin(wdir)

        # Yaw rotation
        cy = np.cos(yaw)
        sy = np.sin(yaw)

        u_yaw = cy * u_fst + sy * v_fst
        v_yaw = -sy * u_fst + cy * v_fst

        # Azimuth rotation
        psi = geom.theta_mesh
        cpsi = np.cos(psi)

        Vax = u_yaw
        Vtn_NR = cpsi * v_yaw

        Vtan = (1 + aprime) * tsr * geom.mu_mesh - Vtn_NR
        phi = np.arctan2(Vax, Vtan)
        aoa = phi - rotor.twist(geom.mu_mesh) - pitch
        aoa = np.clip(aoa, -np.pi / 2, np.pi / 2)

        Cl, Cd = rotor.clcd(geom.mu_mesh, aoa)
        solidity = rotor.solidity(geom.mu_mesh)

        return AerodynamicProperties(
            an=an,
            aprime=aprime,
            solidity=solidity,
            U=U * np.ones(geom.shape),
            wdir=wdir * np.ones(geom.shape),
            Vax=Vax,
            Vtan=Vtan,
            aoa=aoa,
            Cl=Cl,
            Cd=Cd,
        )