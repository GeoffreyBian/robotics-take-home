import json
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Prior:
    """Real-world facts a reconstructed mesh can't tell us: absolute size and material."""

    category: str
    material: str
    longest_m: float
    friction: float
    # exactly one of these; mass wins for thin-walled or multi-material objects
    # where reconstructed volume x density would be badly off
    mass_kg: Optional[float] = None
    density_kg_m3: Optional[float] = None
    note: str = ""

    def __post_init__(self):
        if (self.mass_kg is None) == (self.density_kg_m3 is None):
            raise ValueError(f"{self.category}: set exactly one of mass_kg / density_kg_m3")
        if self.longest_m <= 0 or self.friction < 0:
            raise ValueError(f"{self.category}: longest_m must be > 0 and friction >= 0")


def load_priors(path: str) -> dict[str, Prior]:
    with open(path) as f:
        return {name: Prior(**fields) for name, fields in json.load(f).items()}
