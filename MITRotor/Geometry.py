from typing import Tuple
from numpy.typing import ArrayLike
import numpy as np

__all__ = ["BEMGeometry"]


class BEMGeometry:
    def __init__(self, Nr, Ntheta):
        self.Nr = Nr
        self.Ntheta = Ntheta

        # self.mu = np.linspace(0.0, 0.99999, Nr)
        # self.mu = np.linspace(0.00001, 0.9999999, 26)
        self.mu = np.array([0.04288751, 0.08042134, 0.11795516, 0.15548898, 0.19302281,0.23055663, 0.26809045, 0.30562428, 0.3431581 , 0.38069192,0.41822574, 0.45575957, 0.49329339, 0.53082721, 0.56836104,0.60589486, 0.64342868, 0.6809625 , 0.71849633, 0.75603015,0.79356397, 0.8310978 , 0.86863162, 0.90616544, 0.94369927,0.98123309])
        self.theta = np.linspace(0.0, 2 * np.pi, Ntheta)

        self.theta_mesh, self.mu_mesh = np.meshgrid(self.theta, self.mu)

    @property
    def shape(self):
        return self.Nr, self.Ntheta

    @property
    def dmu(self):
        return self.mu[1] - self.mu[0]

    @property
    def dtheta(self):
        return 2 * np.pi / 158
        # return self.theta[1] - self.theta[0]

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
        # X_azim = 1 / (2 * np.pi) * np.trapz(X, self.theta_mesh, axis=-1)

        theta = self.theta

        dtheta = np.gradient(theta)
        weights = dtheta / np.sum(dtheta)
        X_azim = np.sum(X * weights, axis=1)

        return X_azim

    def rotor_average(self, X: ArrayLike):
        # Takes annulus average quantities and performs rotor average

        # X_rotor = 2 * np.trapz(X * self.mu, self.mu)

        r = self.mu
        # theta = self.theta

        # dr = np.gradient(r)
        # dtheta = np.gradient(theta)

        # area_elements = np.outer(dr * r, dtheta)
        # integrand = area_elements.T * X

        # A = np.pi * (1 - r[0]**2)

        # X_rotor  = np.sum(integrand) / A



        dr = np.gradient(r)
        area_elements = 2 * np.pi * r * dr  # differential area for annular rings
        integrand = X * area_elements

        A = np.pi * (1 - r[0]**2)  # total area of annulus from R1 to R2

        X_rotor =  np.sum(integrand) / A

        return X_rotor