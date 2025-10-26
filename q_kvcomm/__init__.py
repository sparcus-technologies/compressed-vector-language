"""Q-KVComm: Quantized Key-Value Communication for Multi-Agent LLMs"""

from .config import QKVCommConfig
from .quantization import QuantizationEngine
from .calibration import CalibrationModule
from .kv_manager import KVCacheManager
from .integration import QKVCommSystem

__version__ = "1.0.0"
__all__ = [
    "QKVCommConfig",
    "QuantizationEngine", 
    "CalibrationModule",
    "KVCacheManager",
    "QKVCommSystem"
]