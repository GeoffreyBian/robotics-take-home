import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Prior:
    """Real-world facts a reconstructed mesh can't tell us: absolute size and material."""

    category: str
    material: str  # key into materials.json
    longest_m: float
    # set for thin-walled or oversized reconstructions where volume x density would be
    # badly off; otherwise mass comes from the material's density
    mass_kg: Optional[float] = None
    # how mass is spread: solid (default), shell, components, liquid; see physics.py
    mass_model: dict = field(default_factory=lambda: {"type": "solid"})
    # optional joint template: lid, flaps, office_chair; see articulate.py
    articulation: Optional[dict] = None
    note: str = ""

    def __post_init__(self):
        if self.longest_m <= 0:
            raise ValueError(f"{self.category}: longest_m must be > 0")
        if self.mass_kg is not None and self.mass_kg <= 0:
            raise ValueError(f"{self.category}: mass_kg must be > 0")


def load_priors(path: str) -> dict[str, Prior]:
    with open(path) as f:
        return {name: Prior(**fields) for name, fields in json.load(f).items()}
