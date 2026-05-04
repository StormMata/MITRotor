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

        # self.mu = np.linspace(Rhub/R, 0.9999, Nr)

        self.mu_root = Rhub / R

        self.mu_edges = np.linspace(self.mu_root, 1.0, Nr + 1)
        self.mu = 0.5 * (self.mu_edges[:-1] + self.mu_edges[1:])

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
    
    # def annulus_average(self, X: ArrayLike):
    #     X_azim = 1 / (2 * np.pi) * np.trapezoid(X, self.theta_mesh, axis=-1)

    #     return X_azim

    # def rotor_average(self, X: ArrayLike):
    #     # Takes annulus average quantities and performs rotor average

    #     X_rotor = 2 * np.trapezoid(X * self.mu, self.mu)
    #     return X_rotor

    def annulus_average(self, X):
        return np.mean(X, axis=-1)

    @property
    def annulus_area_weights(self):
        return self.mu_edges[1:]**2 - self.mu_edges[:-1]**2

    def rotor_average(self, X):
        w = self.annulus_area_weights

        if X.ndim == 2:
            # sector quantity, shape (Nr, Ntheta)
            return np.sum(X * w[:, None]) / (self.Ntheta * np.sum(w))

        elif X.ndim == 1:
            # annulus quantity, shape (Nr,)
            return np.sum(X * w) / np.sum(w)

        else:
            raise ValueError("Expected X with shape (Nr,) or (Nr, Ntheta).")

    @property
    def dr(self):
        return (self.Radius - self.Hub_radius) / self.Nr

    @property
    def dtheta(self):
        return 2 * np.pi / self.Ntheta