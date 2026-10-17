import numpy as np
from netCDF4 import Dataset


# --------------------- Lightweight Wrapper ---------------------
class MPASVariable(np.ndarray):
    """
    Numpy array subclass with metadata attributes.
    Attributes:
        units : str
        variable_name : str
    """
    def __new__(cls, input_array, variable_name, units):
        obj = np.asarray(input_array).view(cls)
        obj.units = units
        obj.variable_name = variable_name
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.units = getattr(obj, 'units', None)
        self.variable_name = getattr(obj, 'variable_name', None)


# --------------------- Internal Helpers ---------------------
def _get_pressure(mpas):
    if 'pressure' in mpas.variables:
        return mpas['pressure'][0, :, :]
    return mpas['pressure_base'][0, :] + mpas['pressure_p'][0, :]


def _calculate_rh(qv, theta, pressure):
    temp = theta * (pressure / 100000.0) ** 0.286
    es = 6.112 * np.exp(17.67 * (temp - 273.15) / (temp - 29.65)) * 100.0
    e = pressure * qv / (0.622 + qv)
    rh = 100.0 * e / es
    return np.clip(rh, 0, 100)


def _build_mpas_var_map():
    """
    Build variable extraction map lazily.
    Each entry stores a function instead of immediately reading data.
    """
    return {
        # 3D
        'U':   {'func': lambda mpas: mpas['uReconstructZonal'][0, :, :], 'units': r"$\mathrm{m\,s^{-1}}$"},
        'V':   {'func': lambda mpas: mpas['uReconstructMeridional'][0, :, :], 'units': r"$\mathrm{m\,s^{-1}}$"},
        'Tv':  {'func': lambda mpas: mpas['theta'][0, :, :], 'units': r"$\mathrm{K}$"},
        'Qv':  {'func': lambda mpas: mpas['qv'][0, :, :] * 1000.0, 'units': r"$\mathrm{g\,kg^{-1}}$"},
        'Q':   {'func': lambda mpas: mpas['qv'][0, :, :] / (1.0 + mpas['qv'][0, :, :]) * 1000.0, 'units': r"$\mathrm{g\,kg^{-1}}$"},
        'Qc':  {'func': lambda mpas: mpas['qc'][0, :, :] * 1000.0, 'units': r"$\mathrm{g\,kg^{-1}}$"},
        'Qi':  {'func': lambda mpas: mpas['qi'][0, :, :] * 1000.0, 'units': r"$\mathrm{g\,kg^{-1}}$"},
        'Qr':  {'func': lambda mpas: mpas['qr'][0, :, :] * 1000.0, 'units': r"$\mathrm{g\,kg^{-1}}$"},
        'Qs':  {'func': lambda mpas: mpas['qs'][0, :, :] * 1000.0, 'units': r"$\mathrm{g\,kg^{-1}}$"},
        'Qg':  {'func': lambda mpas: mpas['qg'][0, :, :] * 1000.0, 'units': r"$\mathrm{g\,kg^{-1}}$"},
        'RHO': {'func': lambda mpas: mpas['rho'][0, :, :], 'units': r"$\mathrm{kg\,m^{-3}}$"},
        'P':   {'func': lambda mpas: _get_pressure(mpas), 'units': r"$\mathrm{Pa}$"},
        'T':   {'func': lambda mpas: mpas['theta'][0, :, :] * (_get_pressure(mpas) / 100000.0) ** 0.286, 'units': r"$\mathrm{K}$"},
        'TK':  {'func': lambda mpas: mpas['theta'][0, :, :] * (_get_pressure(mpas) / 100000.0) ** 0.286, 'units': r"$\mathrm{K}$"},
        'TC':  {'func': lambda mpas: mpas['theta'][0, :, :] * (_get_pressure(mpas) / 100000.0) ** 0.286 - 273.15, 'units': r"$^\circ\mathrm{C}$"},
        'RH':  {'func': lambda mpas: _calculate_rh(
            mpas['qv'][0, :, :],
            mpas['theta'][0, :, :],
            _get_pressure(mpas)
        ), 'units': r"$\%$"},
        'Z':   {'func': lambda mpas: 0.5 * (mpas['zgrid'][:, :-1] + mpas['zgrid'][:, 1:]), 'units': r"$\mathrm{m}$"},
        'ZH':  {'func': lambda mpas: np.clip(mpas['refl10cm'][0, :, :], 0, 75), 'units': r"$\mathrm{dBZ}$"},

        # 2D
        'LAT':     {'func': lambda mpas: np.degrees(mpas['latCell'][:]), 'units': r"$^\circ$"},
        'LON':     {'func': lambda mpas: np.degrees(mpas['lonCell'][:]), 'units': r"$^\circ$"},
        'terrain': {'func': lambda mpas: mpas['ter'][:], 'units': r"$\mathrm{m}$"},
        'U10':     {'func': lambda mpas: mpas['u10'][0, :], 'units': r"$\mathrm{m\,s^{-1}}$"},
        'V10':     {'func': lambda mpas: mpas['v10'][0, :], 'units': r"$\mathrm{m\,s^{-1}}$"},
        'T2':      {'func': lambda mpas: mpas['t2m'][0, :], 'units': r"$\mathrm{K}$"},
        'Q2':      {'func': lambda mpas: mpas['q2'][0, :] * 1000.0, 'units': r"$\mathrm{g\,kg^{-1}}$"},
        'Ps':      {'func': lambda mpas: mpas['surface_pressure'][0, :], 'units': r"$\mathrm{Pa}$"},
        'RAIN':    {'func': lambda mpas: mpas['rainc'][0, :] + mpas['rainnc'][0, :], 'units': r"$\mathrm{mm}$"},
    }


# --------------------- Public Functions ---------------------
def extract_mpas_variables(mpas_file, variable_name):
    """
    Extract one variable from an MPAS file.
    Returns data as MPASVariable.
    """
    with Dataset(mpas_file, 'r') as mpas:
        mpas_vars = _build_mpas_var_map()
        if variable_name in mpas_vars:
            data = mpas_vars[variable_name]['func'](mpas)
            units = mpas_vars[variable_name]['units']
        elif variable_name in mpas.variables:
            var = mpas[variable_name]
            data = var[0, ...] if 'Time' in var.dimensions else var[:]
            units = getattr(var, 'units', 'unknown')
        else:
            raise ValueError(f"Unsupported or unknown MPAS variable: {variable_name}")
    
    return MPASVariable(data, variable_name=variable_name, units=units)

def extract_mpas_variables_list(mpas_file, variable_names):
    """
    Extract a list of variables from an MPAS file in one file-open.

    Parameters
    ----------
    mpas_file : str
        MPAS NetCDF file path
    variable_names : list[str]
        Variables to extract

    Returns
    -------
    dict[str, MPASVariable]
        Dictionary keyed by canonical variable name
    """
    outputs = {}

    with Dataset(mpas_file, 'r') as mpas:
        mpas_vars = _build_mpas_var_map()

        for variable_name in variable_names:
            if variable_name in mpas_vars:
                data = mpas_vars[variable_name]['func'](mpas)
                units = mpas_vars[variable_name]['units']
            elif variable_name in mpas.variables:
                var = mpas[variable_name]
                data = var[0, ...] if 'Time' in var.dimensions else var[:]
                units = getattr(var, 'units', 'unknown')
            else:
                raise ValueError(f"Unsupported or unknown MPAS variable: {variable_name}")

            outputs[variable_name] = MPASVariable(
                data,
                variable_name=variable_name,
                units=units
            )

    return outputs
