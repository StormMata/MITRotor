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

# -------------------------------------------------------------------------------------------------------------------
class BladeAirfoils:
    @classmethod
    def from_windio(cls, windio: dict, hub_radius, R, N=120):
        blade = windio["components"]["blade"]
        airfoils = windio["airfoils"]
        D = windio["assembly"]["rotor_diameter"]

        airfoil_grid = np.array(blade["outer_shape_bem"]["airfoil_position"]["grid"])
        airfoil_grid_adjusted = (hub_radius + airfoil_grid * (R - hub_radius)) / R

        airfoil_order = blade["outer_shape_bem"]["airfoil_position"]["labels"]

        # Get unique labels in order of first occurrence
        unique_labels = list(dict.fromkeys(airfoil_order))

        # Create the ID mapping (1-based)
        id_map = {label: i + 1 for i, label in enumerate(unique_labels)}

        # Generate the ID array
        airfoil_ids = np.array([id_map[name] for name in airfoil_order])

        airfoils = {x["name"]: Airfoil.from_windio_airfoil(x, R) for x in windio["airfoils"]}

        # print(airfoil_grid_adjusted)

        return cls(D, airfoil_grid_adjusted, airfoil_order, airfoils, airfoil_ids, N=N)

    def __init__(self, D, airfoil_grid, airfoil_order, airfoils, airfoil_ids, N=120):
        self.D = D
        self.airfoil_grid = airfoil_grid
        self.airfoil_ids = airfoil_ids

        aoa_grid = np.linspace(-np.pi, np.pi, N)
        cl = np.array([airfoils[name].Cl(aoa_grid) for name in airfoil_order])
        cd = np.array([airfoils[name].Cd(aoa_grid) for name in airfoil_order])

        self.cl_2d_raw = np.array([airfoils[name].Cl(aoa_grid) for name in airfoil_order])
        self.cd_2d_raw = np.array([airfoils[name].Cd(aoa_grid) for name in airfoil_order])

    def Cl(self, x, inflow):
        x_flat = x.ravel()
        inflow_flat = inflow.ravel()

        # Get the airfoil index at each x location
        airfoil_ids = self.airfoil_id_from_interp(x_flat)

        # Clamp AOA to bounds of the aoa_grid
        aoa_grid = np.linspace(-np.pi, np.pi, self.cl_2d_raw.shape[1])
        inflow_clamped = np.clip(inflow_flat, aoa_grid[0], aoa_grid[-1])

        # Interpolate Cl only along AoA for each selected airfoil
        cl_vals = np.empty_like(inflow_flat)
        for i, (aid, aoa) in enumerate(zip(airfoil_ids, inflow_clamped)):
            cl_vals[i] = np.interp(aoa, aoa_grid, self.cl_2d_raw[aid])

        return cl_vals.reshape(x.shape)

    def Cd(self, x, inflow):
        x_flat = x.ravel()
        inflow_flat = inflow.ravel()

        airfoil_ids = self.airfoil_id_from_interp(x_flat)

        aoa_grid = np.linspace(-np.pi, np.pi, self.cd_2d_raw.shape[1])
        inflow_clamped = np.clip(inflow_flat, aoa_grid[0], aoa_grid[-1])

        cd_vals = np.empty_like(inflow_flat)
        for i, (aid, aoa) in enumerate(zip(airfoil_ids, inflow_clamped)):
            cd_vals[i] = np.interp(aoa, aoa_grid, self.cd_2d_raw[aid])

        return cd_vals.reshape(x.shape)

    def __call__(self, x, inflow):
        return self.Cl(x, inflow), self.Cd(x, inflow)

# -------------------------------------------------------------------------------------------------------
    def airfoil_id_from_interp(self, x):
        """
        Mimic Fortran logic: interpolate floating-point airfoil IDs from grid positions,
        then round and clamp to valid integer indices.
        """

        # Floating-point interpolation of airfoil index
        airfoil_id_float = np.interp(x.ravel(), self.airfoil_grid, self.airfoil_ids)
        
        # Round to nearest and clamp between 0 and len-1 (Python is 0-based)
        airfoil_id_int = np.clip(np.round(airfoil_id_float), 0, len(self.airfoil_grid) - 1).astype(int)

        return airfoil_id_int.reshape(x.shape)

# -------------------------------------------------------------------------------------------------------------------

class RotorDefinition:
    @classmethod
    def from_windio(cls, windio: dict):
        name = windio["name"]

        P_rated = windio["assembly"]["rated_power"]
        hub_height = windio["assembly"]["hub_height"]
        rotorspeed_max = windio["control"]["torque"]["VS_maxspd"]
        tsr_target = windio["control"]["torque"]["tsr"]

        hub_diameter = windio["components"]["hub"]["diameter"]
        hub_radius = hub_diameter / 2
        cone = 0
        blade = windio["components"]["blade"]

        blade_length = blade["outer_shape_bem"]["reference_axis"]["z"]["values"][-1]

        N_blades = windio["assembly"]["number_of_blades"]
        D = windio["assembly"]["rotor_diameter"]

        # Rotor radius adjusted to include cone angle and hub diameter
        # R = blade_length * np.cos(cone) + hub_radius + 0.345
        R = D / 2

        data_twist = blade["outer_shape_bem"]["twist"]
        data_chord = blade["outer_shape_bem"]["chord"]

        blade = windio["components"]["blade"]
        airfoil_grid = np.array(blade["outer_shape_bem"]["airfoil_position"]["grid"])
        airfoil_grid_adjusted = (hub_radius + airfoil_grid * (R - hub_radius)) / R

        # grid including hub center and cone angle
        twist_grid = (hub_radius + np.array(data_twist["grid"]) * (R - hub_radius)) / R
        twist_func_old = interpolate.interp1d(twist_grid, data_twist["values"], fill_value="extrapolate")

        # THIS REDEFINES THE TWIST FUNCTION AS A FUNCTION OF THE AIRFOIL GRID FOR COMPATIBILITY WITH WRF-LES
        twist_at_airfoil_grid = twist_func_old(airfoil_grid_adjusted)
        twist_func = interpolate.interp1d(airfoil_grid_adjusted, twist_at_airfoil_grid, kind='linear', fill_value="extrapolate")

        # grid including hub center and cone angle
        chord_grid = (hub_radius + np.array(data_chord["grid"]) * (R - hub_radius)) / R
        chord_func_old = interpolate.interp1d(chord_grid, data_chord["values"], fill_value="extrapolate")

        # THIS REDEFINES THE CHORD FUNCTION AS A FUNCTION OF THE AIRFOIL GRID FOR COMPATIBILITY WITH WRF-LES
        chord_at_airfoil_grid = chord_func_old(airfoil_grid_adjusted)
        chord_func = interpolate.interp1d(airfoil_grid_adjusted, chord_at_airfoil_grid, kind='linear', fill_value="extrapolate")

        # solidity_func = lambda mu: np.minimum(
        #     N_blades * chord_func(mu) / (2 * np.pi * np.maximum(mu, 0.0001) * R),
        #     1,
        # )

        solidity_func = interpolate.interp1d(airfoil_grid_adjusted, 3 * chord_func(airfoil_grid_adjusted) / (2 * np.pi * R * airfoil_grid_adjusted), kind='linear', fill_value="extrapolate")

        airfoil_func = BladeAirfoils.from_windio(windio, hub_radius, R)

        return cls(
            twist_func,
            twist_func_old,
            chord_func,
            chord_func_old,
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
        twist_func_old,
        chord_func,
        chord_func_old,
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
        self.twist_func_old = twist_func_old
        self.chord_func = chord_func
        self.chord_func_old = chord_func_old
        self.solidity_func = solidity_func
        self.airfoil_func = airfoil_func

    def twist(self, mu):
        return self.twist_func(mu)

    def solidity(self, mu):
        return self.solidity_func(mu)

    def clcd(self, mu, aoa):
        return self.airfoil_func(mu, aoa)
