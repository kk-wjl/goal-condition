from importlib import import_module
from .lafan1 import LAFAN1Dataset

Styles100Dataset = import_module(".100styles", __name__).Styles100Dataset

__all__ = [
    "Styles100Dataset",
    "LAFAN1Dataset",
]
