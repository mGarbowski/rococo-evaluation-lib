from dataclasses import dataclass, asdict
import math
from typing import List
import json

@dataclass
class ValidationResults:
    threshold: List[float]
    far: List[float]
    frr: List[float]

    def __post_init__(self):
        if len(self.threshold) != len(self.far) or len(self.threshold) != len(self.frr):
            raise ValueError("Threshold, FAR, and FRR lists must have the same length.")
        if not all(0 <= x <= 1 for x in self.far + self.frr):
            raise ValueError("FAR and FRR values must be between 0 and 1.")

    @property
    def eer(self) -> float:
        if self.far[0] < self.frr[0]:
            return float('nan')
        
        for far_val, frr_val in zip(self.far, self.frr):
            if far_val <= frr_val:
                return far_val
        
        return float('nan')

    @property
    def eer_threshold(self) -> float:
        if self.far[0] < self.frr[0]:
            return float('nan')

        for thresh, far_val, frr_val in zip(self.threshold, self.far, self.frr):
            if far_val <= frr_val:
                return thresh

        return float('nan')
    
    @property
    def frr_at_far_zero(self) -> float:
        """Value of FRR when FAR reaches zero."""
        for far_val, frr_val in zip(self.far, self.frr):
            if math.isclose(far_val, 0.0, abs_tol=1e-5):
                return frr_val
            
        return float('nan')
    
    @property
    def frr_at_far_zero_threshold(self) -> float:
        """Threshold at which FAR reaches zero."""
        for thresh, far_val in zip(self.threshold, self.far):
            if math.isclose(far_val, 0.0, abs_tol=1e-5):
                return thresh
            
        return float('nan')

    def save_json(self, file_path: str):
        with open(file_path, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load_json(cls, file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(**data)