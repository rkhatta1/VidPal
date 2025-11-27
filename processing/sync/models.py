from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any


@dataclass
class FileOffset:
    """Represents a single file's position in the global timeline."""
    
    file_id: str
    file_path: str
    original_duration: float
    
    # Offset from master (positive = starts after master, negative = starts before)
    offset_seconds: float
    
    # Position in GLOBAL timeline (after normalization)
    global_in_point: float
    global_out_point: float
    
    # Confidence of the sync (correlation peak value, 0-1)
    sync_confidence: float
    
    @property
    def is_master(self) -> bool:
        return self.offset_seconds == 0.0 and self.sync_confidence == 1.0
    
    def contains_global_time(self, global_time: float) -> bool:
        """Check if this file has footage at the given global timestamp."""
        return self.global_in_point <= global_time <= self.global_out_point
    
    def global_to_source_time(self, global_time: float) -> Optional[float]:
        """
        Convert a global timeline timestamp to source file timecode.
        Returns None if the global_time is outside this file's range.
        """
        if not self.contains_global_time(global_time):
            return None
        return global_time - self.global_in_point
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_id": self.file_id,
            "file_path": self.file_path,
            "original_duration": self.original_duration,
            "offset_seconds": self.offset_seconds,
            "global_in_point": self.global_in_point,
            "global_out_point": self.global_out_point,
            "sync_confidence": self.sync_confidence,
            "is_master": self.is_master,
        }


@dataclass
class SyncResult:
    """Result of the sync operation across all files."""
    
    master_file_id: str
    file_offsets: Dict[str, FileOffset]
    
    # Global timeline bounds
    global_start: float
    global_end: float
    
    # Common range where ALL files have footage
    common_start: float
    common_end: float
    
    # Flags
    has_full_overlap: bool = False
    
    def get_offset(self, file_id: str) -> Optional[FileOffset]:
        return self.file_offsets.get(file_id)
    
    @property
    def common_duration(self) -> float:
        return self.common_end - self.common_start
    
    @property
    def global_duration(self) -> float:
        return self.global_end - self.global_start
    
    def get_available_files_at(self, global_time: float) -> List[str]:
        """Return list of file_ids that have footage at given global time."""
        return [
            file_id
            for file_id, offset in self.file_offsets.items()
            if offset.contains_global_time(global_time)
        ]
    
    def get_source_timecode(
        self, file_id: str, global_time: float
    ) -> Optional[float]:
        """Convert global time to source timecode for a specific file."""
        offset = self.file_offsets.get(file_id)
        if offset is None:
            return None
        return offset.global_to_source_time(global_time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "master_file_id": self.master_file_id,
            "file_offsets": {
                fid: fo.to_dict() for fid, fo in self.file_offsets.items()
            },
            "global_start": self.global_start,
            "global_end": self.global_end,
            "common_start": self.common_start,
            "common_end": self.common_end,
            "has_full_overlap": self.has_full_overlap,
            "common_duration": self.common_duration,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyncResult":
        """Reconstruct SyncResult from dictionary (e.g., from DB)."""
        file_offsets = {}
        for fid, fo_data in data["file_offsets"].items():
            file_offsets[fid] = FileOffset(
                file_id=fo_data["file_id"],
                file_path=fo_data["file_path"],
                original_duration=fo_data["original_duration"],
                offset_seconds=fo_data["offset_seconds"],
                global_in_point=fo_data["global_in_point"],
                global_out_point=fo_data["global_out_point"],
                sync_confidence=fo_data["sync_confidence"],
            )
        
        return cls(
            master_file_id=data["master_file_id"],
            file_offsets=file_offsets,
            global_start=data["global_start"],
            global_end=data["global_end"],
            common_start=data["common_start"],
            common_end=data["common_end"],
            has_full_overlap=data.get("has_full_overlap", False),
        )
