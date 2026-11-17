"""Declared lumped heat/reflow surrogate; no fluid simulation or strength test."""
from dataclasses import dataclass
import math


@dataclass
class ThermalState:
    tip_c: float=22.
    joint_c: float=22.
    reflow_seconds: float=0.
    wetted: bool=False
    bonded: bool=False
    unsafe: bool=False

    def advance(self,dt,powered,contact,unsafe_exposure=False):
        if not math.isfinite(dt) or dt<=0:raise ValueError('dt must be positive and finite')
        # A preset 350 C iron approaches temperature over about 25 seconds.
        target,tau=(350.,8.) if powered else (22.,25.)
        self.tip_c=target+(self.tip_c-target)*math.exp(-dt/tau)
        target,tau=(self.tip_c,.30) if contact else (22.,.60)
        self.joint_c=target+(self.joint_c-target)*math.exp(-dt/tau)
        if contact and self.joint_c>=217.:
            self.reflow_seconds+=dt
            if self.reflow_seconds>=.60-1e-9:self.wetted=True
        elif not self.wetted:
            # Separate brief taps do not accumulate a complete wetting interval.
            self.reflow_seconds=0.
        if self.wetted and not contact and self.joint_c<180.:self.bonded=True
        if self.tip_c>=180. and unsafe_exposure:self.unsafe=True
