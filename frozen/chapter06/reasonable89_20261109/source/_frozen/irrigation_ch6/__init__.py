"""Independent final environment for the thesis irrigation application."""
from gymnasium.envs.registration import register, registry

from .env import IrrigationEnv, make_env
from .metrics import evaluate_outcomes
from .interface import IrrigationSpec, IrrigationEncoder, append_preference

__version__ = "1.0.0"
ENV_ID = "IrrigationApplication/Hongmiao93-v1"
if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="irrigation_ch6.env:IrrigationEnv")

__all__ = ["IrrigationEnv", "make_env", "evaluate_outcomes", "ENV_ID",
           "IrrigationSpec", "IrrigationEncoder", "append_preference"]
