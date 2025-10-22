
# utils/caching.py
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_cache_key(*args, **kwargs) -> str:
    """Generate a cache key from arguments."""
    key_data = {
        'args': [str(arg) for arg in args],
        'kwargs': {k: str(v) for k, v in sorted(kwargs.items())}
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_string.encode()).hexdigest()


class Cache:
    """Simple file-based cache for pipeline stages."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, stage: str) -> Optional[Any]:
        """Retrieve cached data."""
        cache_file = self.cache_dir / stage / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    logger.info(f"✅ Cache hit for {stage}/{key[:8]}...")
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        return None
    
    def set(self, key: str, stage: str, data: Any) -> None:
        """Store data in cache."""
        stage_dir = self.cache_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        cache_file = stage_dir / f"{key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"💾 Cached {stage}/{key[:8]}...")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def cached(self, stage: str, key_func: Optional[Callable] = None):
        """Decorator for caching function results."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = get_cache_key(*args, **kwargs)
                
                # Try to get from cache
                cached_result = self.get(cache_key, stage)
                if cached_result is not None:
                    return cached_result
                
                # Compute and cache
                result = func(*args, **kwargs)
                self.set(cache_key, stage, result)
                return result
            return wrapper
        return decorator
