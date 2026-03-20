from typing import Tuple
from numpy.typing import ArrayLike
import numpy as np

__all__ = ["BEMGeometry"]


class BEMGeometry:
    def __init__(self, Nr, Ntheta, R, Rhub):
        self.Nr = Nr
        self.Ntheta = Ntheta
        self.Radius = R
        self.Hub_radius = Rhub

        # This replaces the old method for computing mu consistent with WRF-LES for induction modeling study
        dr = (R - Rhub) / Nr
        r_array = Rhub + dr * (np.arange(Nr) + 0.5)
        self.mu = r_array / R

        self.theta = np.linspace(0.0, 2 * np.pi, Ntheta, endpoint=False)

        self.theta_mesh, self.mu_mesh = np.meshgrid(self.theta, self.mu)

    @property
    def shape(self):
        return self.Nr, self.Ntheta

    @property
    def dmu(self):
        return self.mu[1] - self.mu[0]

    @property
    def dtheta(self):
        return 2 * np.pi / self.Ntheta

    def cartesian(self, yaw: float) -> Tuple[ArrayLike, ...]:
        """
        Returns the grid point locations in cartesian coordinates
        nondimensionialized by rotor radius. Origin is located at hub center.

        Note: effect of yaw angle on grid points is not yet implemented.
        """
        # Probable sign error here.
        X = np.zeros_like(self.mu_mesh)
        Y = self.mu_mesh * np.sin(self.theta_mesh)  # lateral
        Z = self.mu_mesh * np.cos(self.theta_mesh)  # vertical

        return X, Y, Z

    def annulus_average(self, X: ArrayLike):
        theta = self.theta

        dtheta = np.gradient(theta)
        weights = dtheta / np.sum(dtheta)
        X_azim = np.sum(X * weights, axis=1)

        return X_azim

    def rotor_average(self, X: ArrayLike):
        # Takes annulus average quantities and performs rotor average
        r = self.mu

        dr = np.gradient(r)
        area_elements = 2 * np.pi * r * dr  # differential area for annular rings
        integrand = X * area_elements

        A = np.pi * (1 - (self.Hub_radius/self.Radius)**2)

        X_rotor =  np.sum(integrand) / A

        return X_rotor