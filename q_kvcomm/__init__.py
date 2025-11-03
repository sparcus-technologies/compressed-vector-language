"""Q-KVComm: Quantized KV Cache Communication for Efficient LLM Agents"""

from .config import QKVCommConfig
from .quantization import QuantizationEngine
from .calibration import CalibrationModule
from .kv_manager import KVCacheManager
from .integration import QKVCommSystem
from .adaptive_extraction import InformationExtractor, ContextTypeDetector, ExtractedFact
from .memory_manager import MemoryManager, AdaptiveCompressionManager

__version__ = "0.2.0"

__all__ = [
    "QKVCommConfig",
    "QuantizationEngine",
    "CalibrationModule",
    "KVCacheManager",
    "QKVCommSystem",
    "InformationExtractor",
    "ContextTypeDetector",
    "ExtractedFact",
    "MemoryManager",
    "AdaptiveCompressionManager",
]