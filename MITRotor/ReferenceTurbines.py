from pathlib import Path
import yaml
from .RotorDefinition import RotorDefinition
from .SGRE2MW import load_sgre_2mw_from_npz

turbine_model_dir = Path(__file__).parent / "ReferenceTurbines"
fn_IEA22MW = turbine_model_dir / "IEA-22-280-RWT.yaml"
fn_IEA15MW = turbine_model_dir / "IEA-15-240-RWT.yaml"
fn_IEA10MW = turbine_model_dir / "IEA-10-198-RWT.yaml"
fn_IEA3_4MW = turbine_model_dir / "IEA-3.4-130-RWT.yaml"

fn_SGRE2MW = turbine_model_dir / "SGRE_2MW_python.npz"

__all__ = ["IEA22MW", "IEA15MW", "IEA10MW", "IEA3_4MW", "SGRE2MW"]

def IEA22MW() -> RotorDefinition:
    with open(fn_IEA22MW, "r") as f:
        data = yaml.safe_load(f)

    return RotorDefinition.from_windio(data)


def IEA15MW() -> RotorDefinition:
    with open(fn_IEA15MW, "r") as f:
        data = yaml.safe_load(f)

    return RotorDefinition.from_windio(data)


def IEA10MW() -> RotorDefinition:
    with open(fn_IEA10MW, "r") as f:
        data = yaml.safe_load(f)

    return RotorDefinition.from_windio(data)


def IEA3_4MW() -> RotorDefinition:
    with open(fn_IEA3_4MW, "r") as f:
        data = yaml.safe_load(f)

    return RotorDefinition.from_windio(data)

def SGRE2MW() -> RotorDefinition:
    """
    SGRE 2.1MW turbine loaded from a Python-friendly exported NPZ.
    (Export produced by export_sgre_2mw_for_python.m)
    """
    return load_sgre_2mw_from_npz(str(fn_SGRE2MW))