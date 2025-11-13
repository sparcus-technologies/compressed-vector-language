"""
Memory-efficient management of extracted facts and KV caches
Handles caching, compression, and memory-aware transmission
"""

import torch
import pickle
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import OrderedDict


@dataclass
class CacheEntry:
    """Represents a cached extraction or KV cache"""
    key: str
    data: Any
    size_bytes: int
    timestamp: float
    access_count: int
    
    def to_dict(self) -> Dict:
        return {
            'key': self.key,
            'size_bytes': self.size_bytes,
            'timestamp': self.timestamp,
            'access_count': self.access_count
        }


class MemoryManager:
    """
    Production-ready memory manager for Q-KVComm
    Features:
    - LRU cache with size limits
    - Disk persistence
    - Compression-aware memory tracking
    - Adaptive eviction policies
    """
    
    def __init__(
        self,
        max_memory_mb: float = 1024.0,  # 1GB default
        cache_dir: Optional[Path] = None,
        enable_disk_cache: bool = True
    ):
        """
        Args:
            max_memory_mb: Maximum memory to use in MB
            cache_dir: Directory for disk cache
            enable_disk_cache: Whether to persist to disk
        """
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.cache_dir = cache_dir or Path.home() / '.q_kvcomm_cache'
        self.enable_disk_cache = enable_disk_cache
        
        # In-memory cache (LRU)
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_memory_bytes = 0
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'disk_loads': 0,
            'disk_saves': 0
        }
        
        # Create cache directory
        if self.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_hash(self, data: str) -> str:
        """Compute hash for cache key"""
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        try:
            # Try pickle size as approximation
            pickled = pickle.dumps(obj)
            return len(pickled)
        except:
            # Fallback estimate
            if isinstance(obj, str):
                return len(obj.encode())
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(
                    self._estimate_size(k) + self._estimate_size(v)
                    for k, v in obj.items()
                )
            elif isinstance(obj, torch.Tensor):
                return obj.numel() * obj.element_size()
            else:
                return 1024  # Default 1KB estimate
    
    def _evict_if_needed(self, required_bytes: int):
        """Evict entries using LRU policy until enough space"""
        while (self.current_memory_bytes + required_bytes > self.max_memory_bytes
               and len(self.cache) > 0):
            # Evict least recently used (first item in OrderedDict)
            key, entry = self.cache.popitem(last=False)
            self.current_memory_bytes -= entry.size_bytes
            self.stats['evictions'] += 1
            
            # Optionally save to disk
            if self.enable_disk_cache:
                self._save_to_disk(key, entry.data)
    
    def _save_to_disk(self, key: str, data: Any):
        """Save entry to disk"""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            self.stats['disk_saves'] += 1
        except Exception as e:
            print(f"Warning: Failed to save to disk: {e}")
    
    def _load_from_disk(self, key: str) -> Optional[Any]:
        """Load entry from disk"""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                self.stats['disk_loads'] += 1
                return data
        except Exception as e:
            print(f"Warning: Failed to load from disk: {e}")
        return None
    
    def get(self, context: str, prefix: str = "facts") -> Optional[Any]:
        """
        Retrieve cached data
        
        Args:
            context: Context string to hash
            prefix: Cache key prefix (e.g., 'facts', 'kv_cache')
            
        Returns:
            Cached data or None
        """
        cache_key = f"{prefix}_{self._compute_hash(context)}"
        
        # Check in-memory cache
        if cache_key in self.cache:
            entry = self.cache.pop(cache_key)  # Remove
            entry.access_count += 1
            self.cache[cache_key] = entry  # Re-insert at end (most recent)
            self.stats['hits'] += 1
            return entry.data
        
        # Check disk cache
        if self.enable_disk_cache:
            data = self._load_from_disk(cache_key)
            if data is not None:
                # Promote to memory cache
                self.set(context, data, prefix=prefix)
                self.stats['hits'] += 1
                return data
        
        self.stats['misses'] += 1
        return None
    
    def set(self, context: str, data: Any, prefix: str = "facts"):
        """
        Store data in cache
        
        Args:
            context: Context string to hash
            data: Data to cache
            prefix: Cache key prefix
        """
        cache_key = f"{prefix}_{self._compute_hash(context)}"
        
        # Estimate size
        size_bytes = self._estimate_size(data)
        
        # Evict if needed
        self._evict_if_needed(size_bytes)
        
        # Create entry
        import time
        entry = CacheEntry(
            key=cache_key,
            data=data,
            size_bytes=size_bytes,
            timestamp=time.time(),
            access_count=0
        )
        
        # Store in memory
        if cache_key in self.cache:
            # Update existing
            old_entry = self.cache.pop(cache_key)
            self.current_memory_bytes -= old_entry.size_bytes
        
        self.cache[cache_key] = entry
        self.current_memory_bytes += size_bytes
    
    def clear(self):
        """Clear all caches"""
        self.cache.clear()
        self.current_memory_bytes = 0
        
        if self.enable_disk_cache:
            # Clear disk cache
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        hit_rate = (
            self.stats['hits'] / max(1, self.stats['hits'] + self.stats['misses'])
        )
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'current_memory_mb': self.current_memory_bytes / (1024 * 1024),
            'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
            'memory_usage_pct': 100 * self.current_memory_bytes / self.max_memory_bytes,
            'num_entries': len(self.cache)
        }
    
    def optimize_memory(self):
        """
        Optimize memory usage
        - Evict low-access entries
        - Compress data if possible
        """
        # Sort by access count
        entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].access_count
        )
        
        # Evict bottom 20% if memory usage > 80%
        if self.current_memory_bytes > 0.8 * self.max_memory_bytes:
            num_to_evict = max(1, len(entries) // 5)
            
            for i in range(num_to_evict):
                key, entry = entries[i]
                if key in self.cache:
                    self.cache.pop(key)
                    self.current_memory_bytes -= entry.size_bytes
                    self.stats['evictions'] += 1
                    
                    if self.enable_disk_cache:
                        self._save_to_disk(key, entry.data)


class AdaptiveCompressionManager:
    """
    Manages adaptive compression based on memory pressure
    """
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.compression_levels = {
            'none': 1.0,
            'light': 1.5,
            'medium': 2.0,
            'aggressive': 3.0
        }
    
    def get_compression_level(self) -> str:
        """Determine compression level based on memory usage"""
        stats = self.memory_manager.get_stats()
        usage_pct = stats['memory_usage_pct']
        
        if usage_pct < 50:
            return 'none'
        elif usage_pct < 70:
            return 'light'
        elif usage_pct < 85:
            return 'medium'
        else:
            return 'aggressive'
    
    def apply_compression(
        self,
        data: Any,
        level: Optional[str] = None
    ) -> Tuple[Any, Dict]:
        """
        Apply compression to data
        
        Args:
            data: Data to compress
            level: Compression level (auto-detected if None)
            
        Returns:
            (compressed_data, metadata)
        """
        if level is None:
            level = self.get_compression_level()
        
        if level == 'none':
            return data, {'compressed': False, 'level': 'none'}
        
        # For torch tensors, apply quantization
        if isinstance(data, torch.Tensor):
            if level == 'light':
                # FP16
                compressed = data.half()
            elif level == 'medium':
                # INT8
                compressed = data.to(torch.int8)
            else:  # aggressive
                # INT4 (simulate with INT8)
                compressed = torch.clamp(
                    (data * 8).round(), -8, 7
                ).to(torch.int8)
            
            metadata = {
                'compressed': True,
                'level': level,
                'original_dtype': str(data.dtype),
                'original_shape': data.shape,
                'compression_ratio': self.compression_levels[level]
            }
            
            return compressed, metadata
        
        # For other data, just return as-is
        return data, {'compressed': False, 'level': level}