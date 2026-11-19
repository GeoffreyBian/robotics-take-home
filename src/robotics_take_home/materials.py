import json
from dataclasses import dataclass

# what the preview scene's floor and the test gripper's pads are made of
FLOOR = "wood"
GRIPPER_PAD = "rubber"


@dataclass(frozen=True)
class Material:
    name: str
    density: float
    default_friction: float
    specular: float
    shininess: float
    opacity: float = 1.0


class MaterialTable:
    def __init__(self, path: str):
        with open(path) as f:
            raw = json.load(f)
        self.materials = {k: Material(name=k, **v) for k, v in raw["materials"].items()}
        self._pairs = {frozenset(k.split("|")): v for k, v in raw["pairs"].items()}

    def __getitem__(self, name: str) -> Material:
        if name not in self.materials:
            raise KeyError(f"unknown material {name!r}; add it to materials.json")
        return self.materials[name]

    def friction(self, a: str, b: str) -> float:
        """Pair coefficient if we have one, else the less grippy of the two defaults."""
        pair = self._pairs.get(frozenset((a, b)))
        if pair is not None:
            return pair
        return min(self[a].default_friction, self[b].default_friction)
