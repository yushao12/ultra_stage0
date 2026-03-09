from dataclasses import dataclass
from pathlib import Path

import mujoco

from mjlab.entity import Entity, EntityCfg


_BOX_XML = Path(__file__).parent.parent / "asset_zoo" / "object" / "box" / "box.xml"


def build_box_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(_BOX_XML))


@dataclass
class BoxCfg(EntityCfg):
  pos: tuple[float, float, float] = (1.0, 0.0, 0.4)
  rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

  def __post_init__(self):
    self.spec_fn = build_box_spec
    self.init_state.pos = self.pos
    self.init_state.rot = self.rot

  def build(self):
    return Entity(self)
