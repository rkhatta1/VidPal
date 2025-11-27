class SyncError(Exception):
    """Base exception for sync module errors."""
    pass


class NoOverlapError(SyncError):
    """Raised when files have no audio overlap."""
    pass


class ExcessiveGapError(SyncError):
    """Raised when time gap between files exceeds threshold."""
    
    def __init__(self, gap_seconds: float, threshold_seconds: float):
        self.gap_seconds = gap_seconds
        self.threshold_seconds = threshold_seconds
        super().__init__(
            f"Time gap of {gap_seconds:.1f}s exceeds maximum allowed "
            f"threshold of {threshold_seconds:.1f}s"
        )


class LowConfidenceError(SyncError):
    """Raised when sync confidence is too low to be reliable."""
    
    def __init__(self, file_id: str, confidence: float, threshold: float):
        self.file_id = file_id
        self.confidence = confidence
        self.threshold = threshold
        super().__init__(
            f"Sync confidence for '{file_id}' is {confidence:.2f}, "
            f"below threshold {threshold:.2f}. Manual sync may be required."
        )


class AudioExtractionError(SyncError):
    """Raised when audio extraction from a file fails."""
    
    def __init__(self, file_path: str, reason: str):
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"Failed to extract audio from '{file_path}': {reason}")
