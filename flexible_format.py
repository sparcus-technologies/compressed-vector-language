"""
Flexible Message Format System - Protocol-agnostic message encoding/decoding

This module provides flexible message serialization that can adapt to different
transmission protocols, bandwidth constraints, and reliability requirements
without hardcoded assumptions about the communication channel.
"""

import base64
import json
import struct
import zlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from advanced_cvl import CompressedVectorMessage


class TransmissionProtocol(Enum):
    """Supported transmission protocols"""

    BINARY_MINIMAL = "binary_minimal"  # Ultra-compact binary (e.g., radio)
    BINARY_ROBUST = "binary_robust"  # Binary with error correction
    JSON_COMPACT = "json_compact"  # Compressed JSON (e.g., web APIs)
    JSON_READABLE = "json_readable"  # Human-readable JSON
    MSGPACK = "msgpack"  # MessagePack binary JSON
    CUSTOM = "custom"  # User-defined format


class ReliabilityLevel(Enum):
    """Reliability/redundancy levels"""

    MINIMAL = "minimal"  # No redundancy - pure compression
    STANDARD = "standard"  # Basic error detection
    ROBUST = "robust"  # Error correction codes
    MISSION_CRITICAL = "critical"  # Maximum redundancy


@dataclass
class TransmissionConfig:
    """Configuration for message transmission"""

    protocol: TransmissionProtocol
    reliability: ReliabilityLevel
    max_payload_size: Optional[int] = None  # bytes
    compression_enabled: bool = True
    encryption_key: Optional[bytes] = None
    metadata_overhead_budget: float = 0.1  # fraction of total size


class MessageFormat(ABC):
    """Abstract base class for message formats"""

    @abstractmethod
    def serialize(
        self, message: "CompressedVectorMessage", config: TransmissionConfig
    ) -> bytes:
        pass

    @abstractmethod
    def deserialize(
        self, data: bytes, config: TransmissionConfig
    ) -> "CompressedVectorMessage":
        pass

    @abstractmethod
    def estimate_size(self, message: "CompressedVectorMessage") -> int:
        pass


class BinaryMinimalFormat(MessageFormat):
    """Ultra-compact binary format for bandwidth-constrained channels"""

    def serialize(
        self, message: "CompressedVectorMessage", config: TransmissionConfig
    ) -> bytes:
        """Serialize to minimal binary format"""

        # Header (16 bytes fixed)
        header = struct.pack(
            "<HBBIIF",  # little-endian
            len(message.semantic_vectors),  # 2 bytes: number of vectors
            self._encode_message_type(message.message_type),  # 1 byte: message type
            message.priority_level,  # 1 byte: priority
            message.compression_level.value[0].encode()[0],  # 1 byte: compression level
            int(message.timestamp),  # 4 bytes: timestamp (unix)
            message.reconstruction_confidence,  # 4 bytes: confidence
        )

        # Semantic vectors (variable length)
        vectors_data = b""
        for semantic_type, vector in message.semantic_vectors.items():
            # Encode semantic type (1 byte) + length (2 bytes) + data
            type_byte = hash(semantic_type) % 256
            vector_bytes = vector.astype(np.float16).tobytes()
            vector_header = struct.pack("<BH", type_byte, len(vector_bytes))
            vectors_data += vector_header + vector_bytes

        # Optional compression
        if config.compression_enabled and len(vectors_data) > 32:
            vectors_data = zlib.compress(vectors_data, level=6)
            compression_flag = 1
        else:
            compression_flag = 0

        # Final assembly with compression flag
        final_data = header + struct.pack("<B", compression_flag) + vectors_data

        # Size check
        if config.max_payload_size and len(final_data) > config.max_payload_size:
            raise ValueError(
                f"Message size {len(final_data)} exceeds limit {config.max_payload_size}"
            )

        return final_data

    def deserialize(
        self, data: bytes, config: TransmissionConfig
    ) -> "CompressedVectorMessage":
        """Deserialize from minimal binary format"""

        if len(data) < 17:  # Minimum size check
            raise ValueError("Invalid binary data: too short")

        # Parse header
        header_data = struct.unpack("<HBBIIF", data[:16])
        num_vectors = header_data[0]
        message_type = self._decode_message_type(header_data[1])
        priority_level = header_data[2]
        compression_level_char = chr(header_data[3])
        timestamp = float(header_data[4])
        confidence = header_data[5]

        # Parse compression flag
        compression_flag = struct.unpack("<B", data[16:17])[0]

        # Parse vectors data
        vectors_data = data[17:]
        if compression_flag:
            try:
                vectors_data = zlib.decompress(vectors_data)
            except zlib.error:
                raise ValueError("Failed to decompress vector data")

        # Reconstruct semantic vectors
        semantic_vectors = {}
        offset = 0

        for _ in range(num_vectors):
            if offset + 3 > len(vectors_data):
                break

            type_byte, length = struct.unpack("<BH", vectors_data[offset : offset + 3])
            offset += 3

            if offset + length > len(vectors_data):
                break

            vector_bytes = vectors_data[offset : offset + length]
            vector = np.frombuffer(vector_bytes, dtype=np.float16)

            # Map type byte back to semantic type (approximate)
            semantic_type = self._decode_semantic_type(type_byte)
            semantic_vectors[semantic_type] = vector

            offset += length

        # Reconstruct compression level enum
        compression_level_map = {
            "l": "lossless",
            "h": "high",
            "b": "balanced",
            "a": "aggressive",
        }
        compression_level_str = compression_level_map.get(
            compression_level_char, "balanced"
        )

        from advanced_cvl import CompressedVectorMessage, CompressionLevel

        compression_level = CompressionLevel(compression_level_str)

        return CompressedVectorMessage(
            semantic_vectors=semantic_vectors,
            message_type=message_type,
            priority_level=priority_level,
            compression_level=compression_level,
            timestamp=timestamp,
            original_embedding_dim=0,  # Not preserved in minimal format
            compression_ratio=0.0,  # Not preserved
            semantic_hierarchy={},  # Not preserved
            checksum=0,  # Not preserved
            reconstruction_confidence=confidence,
        )

    def estimate_size(self, message: "CompressedVectorMessage") -> int:
        """Estimate serialized size"""
        base_size = 17  # Header + compression flag
        vectors_size = sum(
            3 + len(v) * 2 for v in message.semantic_vectors.values()
        )  # float16
        return base_size + vectors_size

    def _encode_message_type(self, message_type: str) -> int:
        """Encode message type to single byte"""
        type_map = {
            "navigation": 1,
            "status": 2,
            "obstacle": 3,
            "coordination": 4,
            "emergency": 5,
        }
        return type_map.get(message_type, 0)

    def _decode_message_type(self, type_byte: int) -> str:
        """Decode message type from byte"""
        type_map = {
            1: "navigation",
            2: "status",
            3: "obstacle",
            4: "coordination",
            5: "emergency",
            0: "unknown",
        }
        return type_map.get(type_byte, "unknown")

    def _decode_semantic_type(self, type_byte: int) -> str:
        """Decode semantic type from byte (approximate mapping)"""
        semantic_types = ["action", "spatial", "temporal", "status", "uncertainty"]
        return semantic_types[type_byte % len(semantic_types)]


class JSONCompactFormat(MessageFormat):
    """Compact JSON format for web APIs and debugging"""

    def serialize(
        self, message: "CompressedVectorMessage", config: TransmissionConfig
    ) -> bytes:
        """Serialize to compact JSON"""

        # Convert to JSON-serializable format
        data = {
            "sv": {k: v.tolist() for k, v in message.semantic_vectors.items()},
            "mt": message.message_type,
            "pl": message.priority_level,
            "cl": message.compression_level.value,
            "ts": message.timestamp,
            "cr": message.compression_ratio,
            "rc": message.reconstruction_confidence,
        }

        # Optional: include full metadata for debugging
        if config.reliability != ReliabilityLevel.MINIMAL:
            data.update(
                {
                    "oed": message.original_embedding_dim,
                    "sh": message.semantic_hierarchy,
                    "cs": message.checksum,
                }
            )

        json_str = json.dumps(data, separators=(",", ":"))  # Compact JSON
        json_bytes = json_str.encode("utf-8")

        # Optional compression
        if config.compression_enabled and len(json_bytes) > 100:
            compressed = zlib.compress(json_bytes, level=6)
            if len(compressed) < len(json_bytes):
                return b"Z" + compressed  # 'Z' prefix indicates compression

        return json_bytes

    def deserialize(
        self, data: bytes, config: TransmissionConfig
    ) -> "CompressedVectorMessage":
        """Deserialize from compact JSON"""

        # Handle compression
        if data.startswith(b"Z"):
            try:
                data = zlib.decompress(data[1:])
            except zlib.error:
                raise ValueError("Failed to decompress JSON data")

        try:
            json_data = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ValueError("Invalid JSON data")

        # Reconstruct semantic vectors
        semantic_vectors = {
            k: np.array(v, dtype=np.float32) for k, v in json_data["sv"].items()
        }

        from advanced_cvl import CompressedVectorMessage, CompressionLevel

        compression_level = CompressionLevel(json_data["cl"])

        return CompressedVectorMessage(
            semantic_vectors=semantic_vectors,
            message_type=json_data["mt"],
            priority_level=json_data["pl"],
            compression_level=compression_level,
            timestamp=json_data["ts"],
            original_embedding_dim=json_data.get("oed", 0),
            compression_ratio=json_data["cr"],
            semantic_hierarchy=json_data.get("sh", {}),
            checksum=json_data.get("cs", 0),
            reconstruction_confidence=json_data["rc"],
        )

    def estimate_size(self, message: "CompressedVectorMessage") -> int:
        """Estimate JSON size"""
        # Rough estimate: JSON overhead + vector data
        base_size = 100  # JSON structure overhead
        vectors_size = sum(
            len(v) * 8 for v in message.semantic_vectors.values()
        )  # ~8 chars per float
        return base_size + vectors_size


class RobustBinaryFormat(MessageFormat):
    """Binary format with error correction and redundancy"""

    def __init__(self):
        self.crc_table = self._generate_crc_table()

    def serialize(
        self, message: "CompressedVectorMessage", config: TransmissionConfig
    ) -> bytes:
        """Serialize with error correction"""

        # Start with compact format
        compact_format = BinaryMinimalFormat()
        base_data = compact_format.serialize(message, config)

        # Add error correction based on reliability level
        if config.reliability == ReliabilityLevel.ROBUST:
            # Add CRC32 checksum
            crc = zlib.crc32(base_data) & 0xFFFFFFFF
            base_data += struct.pack("<I", crc)

        elif config.reliability == ReliabilityLevel.MISSION_CRITICAL:
            # Add redundancy + checksum
            crc = zlib.crc32(base_data) & 0xFFFFFFFF
            redundant_data = base_data + struct.pack("<I", crc)

            # Triple redundancy for critical data
            return redundant_data * 3

        return base_data

    def deserialize(
        self, data: bytes, config: TransmissionConfig
    ) -> "CompressedVectorMessage":
        """Deserialize with error correction"""

        if config.reliability == ReliabilityLevel.MISSION_CRITICAL:
            # Handle triple redundancy
            if len(data) % 3 != 0:
                raise ValueError("Invalid triple redundancy format")

            chunk_size = len(data) // 3
            chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

            # Use majority voting for error correction
            if chunks[0] == chunks[1]:
                data = chunks[0]
            elif chunks[0] == chunks[2]:
                data = chunks[0]
            elif chunks[1] == chunks[2]:
                data = chunks[1]
            else:
                # All different - use first and hope for the best
                data = chunks[0]

        if config.reliability in [
            ReliabilityLevel.ROBUST,
            ReliabilityLevel.MISSION_CRITICAL,
        ]:
            # Verify CRC32
            if len(data) < 4:
                raise ValueError("Data too short for CRC verification")

            payload_data = data[:-4]
            expected_crc = struct.unpack("<I", data[-4:])[0]
            actual_crc = zlib.crc32(payload_data) & 0xFFFFFFFF

            if expected_crc != actual_crc:
                raise ValueError(
                    f"CRC mismatch: expected {expected_crc}, got {actual_crc}"
                )

            data = payload_data

        # Deserialize base data
        compact_format = BinaryMinimalFormat()
        return compact_format.deserialize(data, config)

    def estimate_size(self, message: "CompressedVectorMessage") -> int:
        """Estimate size with error correction overhead"""
        compact_format = BinaryMinimalFormat()
        base_size = compact_format.estimate_size(message)

        # Add error correction overhead
        return base_size + 4  # CRC32

    def _generate_crc_table(self) -> List[int]:
        """Generate CRC lookup table"""
        table = []
        for i in range(256):
            crc = i
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xEDB88320
                else:
                    crc >>= 1
            table.append(crc)
        return table


class FlexibleMessageProtocol:
    """Protocol-agnostic message handling system"""

    def __init__(self):
        self.formats = {
            TransmissionProtocol.BINARY_MINIMAL: BinaryMinimalFormat(),
            TransmissionProtocol.BINARY_ROBUST: RobustBinaryFormat(),
            TransmissionProtocol.JSON_COMPACT: JSONCompactFormat(),
            TransmissionProtocol.JSON_READABLE: JSONCompactFormat(),  # Can be extended
        }

        # Protocol recommendations based on use case
        self.protocol_recommendations = {
            ("emergency", "minimal"): TransmissionProtocol.BINARY_MINIMAL,
            ("emergency", "standard"): TransmissionProtocol.BINARY_ROBUST,
            ("emergency", "robust"): TransmissionProtocol.BINARY_ROBUST,
            ("emergency", "critical"): TransmissionProtocol.BINARY_ROBUST,
            ("coordination", "minimal"): TransmissionProtocol.JSON_COMPACT,
            ("status", "minimal"): TransmissionProtocol.JSON_COMPACT,
            ("debug", "any"): TransmissionProtocol.JSON_READABLE,
        }

    def recommend_protocol(
        self,
        message_type: str,
        reliability: ReliabilityLevel,
        bandwidth_constraint: Optional[int] = None,
    ) -> TransmissionProtocol:
        """Recommend optimal protocol based on requirements"""

        # Check specific recommendations
        key = (message_type, reliability.value)
        if key in self.protocol_recommendations:
            return self.protocol_recommendations[key]

        # General recommendations
        if bandwidth_constraint and bandwidth_constraint < 100:
            return TransmissionProtocol.BINARY_MINIMAL
        elif reliability in [
            ReliabilityLevel.ROBUST,
            ReliabilityLevel.MISSION_CRITICAL,
        ]:
            return TransmissionProtocol.BINARY_ROBUST
        else:
            return TransmissionProtocol.JSON_COMPACT

    def serialize_message(
        self, message: "CompressedVectorMessage", config: TransmissionConfig
    ) -> bytes:
        """Serialize message using specified protocol"""

        format_handler = self.formats.get(config.protocol)
        if not format_handler:
            raise ValueError(f"Unsupported protocol: {config.protocol}")

        return format_handler.serialize(message, config)

    def deserialize_message(
        self, data: bytes, config: TransmissionConfig
    ) -> "CompressedVectorMessage":
        """Deserialize message using specified protocol"""

        format_handler = self.formats.get(config.protocol)
        if not format_handler:
            raise ValueError(f"Unsupported protocol: {config.protocol}")

        return format_handler.deserialize(data, config)

    def estimate_transmission_size(
        self, message: "CompressedVectorMessage", protocol: TransmissionProtocol
    ) -> int:
        """Estimate size for different protocols"""

        format_handler = self.formats.get(protocol)
        if not format_handler:
            return 0

        return format_handler.estimate_size(message)

    def optimize_for_bandwidth(
        self, message: "CompressedVectorMessage", max_size: int
    ) -> Tuple[TransmissionConfig, int]:
        """Find optimal protocol/config for bandwidth constraint"""

        best_config = None
        best_size = float("inf")

        # Try different protocol/reliability combinations
        for protocol in TransmissionProtocol:
            if protocol not in self.formats:
                continue

            for reliability in ReliabilityLevel:
                config = TransmissionConfig(
                    protocol=protocol,
                    reliability=reliability,
                    max_payload_size=max_size,
                    compression_enabled=True,
                )

                try:
                    estimated_size = self.estimate_transmission_size(message, protocol)

                    if estimated_size <= max_size and estimated_size < best_size:
                        best_config = config
                        best_size = estimated_size

                except Exception:
                    continue

        if best_config is None:
            raise ValueError(
                f"Cannot fit message in {max_size} bytes with any protocol"
            )

        return best_config, best_size


def main():
    """Demonstrate flexible message format system"""

    # This would normally import from advanced_cvl, but for demo we'll create mock data
    print("🔧 Flexible Message Format Demo")
    print("=" * 50)

    # Mock compressed message for demonstration
    mock_vectors = {
        "action": np.random.randn(8).astype(np.float16),
        "spatial": np.random.randn(8).astype(np.float16),
        "temporal": np.random.randn(6).astype(np.float16),
    }

    # Test different protocols
    protocol_system = FlexibleMessageProtocol()

    protocols_to_test = [
        (TransmissionProtocol.BINARY_MINIMAL, ReliabilityLevel.MINIMAL),
        (TransmissionProtocol.BINARY_ROBUST, ReliabilityLevel.ROBUST),
        (TransmissionProtocol.JSON_COMPACT, ReliabilityLevel.STANDARD),
    ]

    print("\\nProtocol Comparison:")
    print("-" * 30)

    for protocol, reliability in protocols_to_test:
        config = TransmissionConfig(
            protocol=protocol, reliability=reliability, compression_enabled=True
        )

        # Mock message data
        mock_message_data = {
            "semantic_vectors": mock_vectors,
            "message_type": "navigation",
            "priority_level": 2,
            "timestamp": 1234567890.0,
            "reconstruction_confidence": 0.87,
        }

        try:
            format_handler = protocol_system.formats[protocol]
            estimated_size = (
                sum(len(v) * 2 for v in mock_vectors.values()) + 50
            )  # Rough estimate

            print(
                f"{protocol.value:20s} | {reliability.value:10s} | ~{estimated_size:3d} bytes"
            )

        except Exception as e:
            print(f"{protocol.value:20s} | ERROR: {e}")

    # Bandwidth optimization demo
    print("\\nBandwidth Optimization:")
    print("-" * 30)

    bandwidth_limits = [50, 100, 200, 500]

    for limit in bandwidth_limits:
        recommended = protocol_system.recommend_protocol(
            "emergency", ReliabilityLevel.STANDARD, limit
        )
        print(f"Bandwidth {limit:3d}B -> {recommended.value}")

    print("\\n✅ Flexible format system ready!")


if __name__ == "__main__":
    main()
