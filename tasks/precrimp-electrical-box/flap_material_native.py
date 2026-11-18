"""Passive elastic-plastic sheet hinges for native Isaac forming trials.

Loaded only after Isaac starts. Plastic rest angles evolve from measured strain
through return mapping. No desired fold angle, robot stage or task goal enters
this material model. The drive represents elastic recovery, not an actuator.
"""
import torch
from isaaclab.actuators import ImplicitActuator,ImplicitActuatorCfg
from isaaclab.utils import configclass
import install_physics as physics


def return_map(joint_pos,rest,accumulated,dissipation,stiffness,yield_moments,hardening):
    trial=stiffness*(joint_pos-rest)
    radius=yield_moments+hardening*accumulated
    increment=torch.clamp(torch.abs(trial)-radius,min=0.)/(stiffness+hardening)
    return (rest+torch.sign(trial)*increment,accumulated+increment,
            dissipation+yield_moments*increment)


class PlasticFlapActuator(ImplicitActuator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        moments=physics.fold_moments(stress=250e6*self.cfg.yield_scale)
        self.yield_moments=torch.tensor([[moments[n] for n in self.joint_names]],device=self._device)
        self.plastic_rest=None;self.accumulated=None;self.dissipation=None;self.updates=0

    def reset(self,env_ids=None):
        if self.plastic_rest is not None:
            ids=slice(None) if env_ids is None else env_ids
            self.plastic_rest[ids]=0.;self.accumulated[ids]=0.;self.dissipation[ids]=0.

    def compute(self,control_action,joint_pos,joint_vel):
        if self.plastic_rest is None:
            self.plastic_rest=torch.zeros_like(joint_pos)
            self.accumulated=torch.zeros_like(joint_pos);self.dissipation=torch.zeros_like(joint_pos)
        self.plastic_rest,self.accumulated,self.dissipation=return_map(
            joint_pos,self.plastic_rest,self.accumulated,self.dissipation,
            self.stiffness,self.yield_moments,self.cfg.hardening)
        self.updates+=1
        control_action.joint_positions=self.plastic_rest
        control_action.joint_velocities=torch.zeros_like(joint_vel)
        control_action.joint_efforts=torch.zeros_like(joint_vel)
        return super().compute(control_action,joint_pos,joint_vel)


@configclass
class PlasticFlapActuatorCfg(ImplicitActuatorCfg):
    class_type:type=PlasticFlapActuator
    yield_scale:float=1.
    hardening:float=.05
