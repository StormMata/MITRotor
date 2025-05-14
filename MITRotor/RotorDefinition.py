import numpy as np
from scipy import interpolate


class Airfoil:
    @classmethod
    def from_windio_airfoil(cls, airfoil: dict, R: float):
        assert len(airfoil["polars"]) == 1

        grid = np.array(airfoil["polars"][0]["c_l"]["grid"])
        return cls(
            airfoil["name"],
            grid,
            airfoil["polars"][0]["c_l"]["values"],
            airfoil["polars"][0]["c_d"]["values"],
        )

    def __init__(self, name, grid, cl, cd):
        self.name = name
        self.Cl_interp = interpolate.interp1d(grid, cl, fill_value="extrapolate")
        self.Cd_interp = interpolate.interp1d(grid, cd, fill_value="extrapolate")

    def __repr__(self):
        return f"Airfoil: {self.name}"

    def Cl(self, angle):
        return self.Cl_interp(angle)

    def Cd(self, angle):
        return self.Cd_interp(angle)


class BladeAirfoils:
    @classmethod
    def from_windio(cls, windio: dict, hub_radius, R, N=120):
        blade = windio["components"]["blade"]
        airfoils = windio["airfoils"]
        D = windio["assembly"]["rotor_diameter"]

        airfoil_grid = np.array(blade["outer_shape_bem"]["airfoil_position"]["grid"])
        # airfoil_grid_adjusted = (hub_radius + airfoil_grid * (R - hub_radius)) / R
        airfoil_grid_adjusted = (hub_radius + airfoil_grid * (R - hub_radius)) / R
        airfoil_order = blade["outer_shape_bem"]["airfoil_position"]["labels"]

        airfoils = {x["name"]: Airfoil.from_windio_airfoil(x, R) for x in windio["airfoils"]}

        # print(airfoil_grid)
        # print(airfoil_grid_adjusted)
        # print(airfoil_order)
        # print(airfoils)

        return cls(D, airfoil_grid_adjusted, airfoil_order, airfoils, N=N)

    def __init__(self, D, airfoil_grid, airfoil_order, airfoils, N=120):
        self.D = D

        aoa_grid = np.linspace(-np.pi, np.pi, N)
        cl = np.array([airfoils[name].Cl(aoa_grid) for name in airfoil_order])
        cd = np.array([airfoils[name].Cd(aoa_grid) for name in airfoil_order])

        self.cl_interp = interpolate.RectBivariateSpline(airfoil_grid, aoa_grid, cl)
        self.cd_interp = interpolate.RectBivariateSpline(airfoil_grid, aoa_grid, cd)

    #     # Save for clamping
    #     self.airfoil_grid = airfoil_grid
    #     self.aoa_grid = aoa_grid

    #     # Interpolators: axes must be (airfoil_grid, aoa_grid)
    #     self.cl_interp = interpolate.RegularGridInterpolator(
    #         (airfoil_grid, aoa_grid), cl, method='linear', bounds_error=False, fill_value=None
    #     )

    #     self.cd_interp = interpolate.RegularGridInterpolator(
    #         (airfoil_grid, aoa_grid), cd, method='linear', bounds_error=False, fill_value=None
    #     )

    # def Cl(self, x, inflow):
    #     x = np.asarray(x)
    #     inflow = np.asarray(inflow)

    #     # Flatten inputs and check shapes
    #     if x.shape != inflow.shape:
    #         raise ValueError(f"x and inflow must have the same shape. Got {x.shape} and {inflow.shape}")

    #     # Clamp to valid ranges
    #     x = np.clip(x, self.airfoil_grid[0], self.airfoil_grid[-1])
    #     inflow = np.clip(inflow, self.aoa_grid[0], self.aoa_grid[-1])

    #     # Stack into (N, 2)
    #     points = np.stack((x, inflow), axis=-1)
    #     return self.cl_interp(points)

    # def Cd(self, x, inflow):
    #     x = np.asarray(x)
    #     inflow = np.asarray(inflow)

    #     if x.shape != inflow.shape:
    #         raise ValueError(f"x and inflow must have the same shape. Got {x.shape} and {inflow.shape}")

    #     x = np.clip(x, self.airfoil_grid[0], self.airfoil_grid[-1])
    #     inflow = np.clip(inflow, self.aoa_grid[0], self.aoa_grid[-1])

    #     points = np.stack((x, inflow), axis=-1)
    #     return self.cd_interp(points)

    def Cl(self, x, inflow):
        return self.cl_interp(x, inflow, grid=False)

    def Cd(self, x, inflow):
        return self.cd_interp(x, inflow, grid=False)

    def __call__(self, x, inflow):
        return self.Cl(x, inflow), self.Cd(x, inflow)


class RotorDefinition:
    @classmethod
    def from_windio(cls, windio: dict):
        name = windio["name"]

        P_rated = windio["assembly"]["rated_power"]
        hub_height = windio["assembly"]["hub_height"]
        rotorspeed_max = windio["control"]["torque"]["VS_maxspd"]
        tsr_target = windio["control"]["torque"]["tsr"]

        hub_diameter = windio["components"]["hub"]["diameter"]
        hub_radius = 2.4
        cone = 0
        blade = windio["components"]["blade"]

        blade_length = blade["outer_shape_bem"]["reference_axis"]["z"]["values"][-1]

        N_blades = windio["assembly"]["number_of_blades"]
        D = windio["assembly"]["rotor_diameter"]

        # Rotor radius adjusted to include cone angle and hub diameter
        # R = blade_length * np.cos(cone) + hub_radius + 0.345
        R = 99.5


        data_twist = blade["outer_shape_bem"]["twist"]
        data_chord = blade["outer_shape_bem"]["chord"]

        # grid including hub center and cone angle
        twist_grid = (hub_radius + np.array(data_twist["grid"]) * (R - hub_radius)) / R
        twist_func = interpolate.interp1d(twist_grid, data_twist["values"], fill_value="extrapolate")
        # grid including hub center and cone angle
        chord_grid = (hub_radius + np.array(data_chord["grid"]) * (R - hub_radius)) / R
        chord_func = interpolate.interp1d(chord_grid, data_chord["values"], fill_value="extrapolate")

        solidity_func = lambda mu: np.minimum(
            N_blades * chord_func(mu) / (2 * np.pi * np.maximum(mu, 0.0001) * R),
            1,
        )

        airfoil_func = BladeAirfoils.from_windio(windio, hub_radius, R)

        return cls(
            twist_func,
            chord_func,
            solidity_func,
            airfoil_func,
            N_blades,
            R,
            P_rated,
            rotorspeed_max,
            hub_height,
            tsr_target,
            hub_radius,
            name=name,
        )

    def __init__(
        self,
        twist_func,
        chord_func,
        solidity_func,
        airfoil_func,
        N_blades,
        R,
        P_rated,
        rotorspeed_max,
        hub_height,
        tsr_target,
        hub_radius,
        name=None,
    ):
        self.name = name
        self.N_blades = N_blades

        self.R = R
        self.P_rated = P_rated
        self.rotorspeed_max = rotorspeed_max
        self.hub_height = hub_height
        self.tsr_target = tsr_target
        self.hub_radius = hub_radius

        self.twist_func = twist_func
        self.chord_func = chord_func
        self.solidity_func = solidity_func
        self.airfoil_func = airfoil_func

    def twist(self, mu):
        t = np.array([ 0.20941403,  0.20958612,  0.20556226,  0.18438513,  0.15035212,0.120011  ,  0.1000569 ,  0.08661511,  0.07561567,  0.06561391,0.05589715,  0.04558797,  0.03434241,  0.0224164 ,  0.01017596,-0.0020152 , -0.01379339, -0.02481012, -0.03475113, -0.04312853,-0.04948717, -0.05347092, -0.05373307, -0.04736611, -0.02696904,-0.01818005])
        
        return t[:, np.newaxis] * np.ones_like(mu)
        # return self.twist_func(mu)

    def solidity(self, mu):
        chord = np.array([4.649046,4.844576,5.216862,5.647547,5.931514,5.96353,5.812206,5.569981,5.273268,4.94348,4.598646,4.251929,3.911027,3.581805,3.269938,2.978544,2.70897,2.462014,2.237137,2.035348,1.855479,1.691271,1.526768,1.308778,0.7955106,0.5734])

        rOverR = np.array([0.04288751, 0.08042134, 0.11795516, 0.15548898, 0.19302281,0.23055663, 0.26809045, 0.30562428, 0.3431581 , 0.38069192,0.41822574, 0.45575957, 0.49329339, 0.53082721, 0.56836104,0.60589486, 0.64342868, 0.6809625 , 0.71849633, 0.75603015,0.79356397, 0.8310978 , 0.86863162, 0.90616544, 0.94369927,0.98123309])
        
        # rOverR = mu

        s = 3 * chord / (2 * np.pi * rOverR * 99.5)

        # print(s)

        # return s


        # return 3*158 * np.ones((26, 158))
        return s[:, np.newaxis] * np.ones((26, 158))
        # return self.solidity_func(mu)

    def clcd(self, mu, aoa):
        return self.airfoil_func(mu, aoa)
