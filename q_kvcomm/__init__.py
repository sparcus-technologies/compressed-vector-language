"""Q-KVComm: Quantized Key-Value Communication for Multi-Agent LLMs"""

from .calibration import CalibrationModule
from .config import QKVCommConfig
from .entity_extraction import EntityExtractor
from .integration import QKVCommSystem
from .kv_manager import KVCacheManager
from .quantization import QuantizationEngine

__version__ = "1.0.0"
__all__ = [
    "QKVCommConfig",
    "QuantizationEngine",
    "CalibrationModule",
    "KVCacheManager",
    "QKVCommSystem",
    "EntityExtractor",
]
