from dataclasses import dataclass

import numpy as np


@dataclass
class SurgicalReachTarget:
    name: str
    position: np.ndarray
    tolerance: float = 0.006
    hold_time: float = 0.35


class SurgicalTargetSequence:
    def __init__(self, targets):
        if not targets:
            raise ValueError("SurgicalTargetSequence requires at least one target")
        self.targets = list(targets)
        self.index = 0
        self._inside_since = None
        self.completed = False

    @property
    def current(self):
        return self.targets[self.index]

    def update(self, tip_pos, sim_time):
        if self.completed:
            return True

        error = float(np.linalg.norm(self.current.position - tip_pos))
        inside = error <= self.current.tolerance

        if inside and self._inside_since is None:
            self._inside_since = sim_time
        elif not inside:
            self._inside_since = None

        if self._inside_since is not None and sim_time - self._inside_since >= self.current.hold_time:
            self.index += 1
            self._inside_since = None
            if self.index >= len(self.targets):
                self.completed = True
                self.index = len(self.targets) - 1
                return True

        return False

    def progress_text(self):
        return f"{self.index + 1}/{len(self.targets)}"
