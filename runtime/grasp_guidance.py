"""Pose-only human collection advice. No simulator, controller or action API."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    def rotation(self) -> np.ndarray:
        q=np.asarray(self.quaternion_wxyz,dtype=float)
        if not np.isfinite(q).all() or np.linalg.norm(q)<1e-12:
            raise ValueError('Invalid orientation')
        w,x,y,z=q/np.linalg.norm(q)
        return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                         [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                         [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


@dataclass(frozen=True)
class SafetyConstraints:
    minimum_clearance_mm: tuple[tuple[str,float], ...] = ()
    forbidden_contacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraspGuidanceConfig:
    reference_type: str
    reference_name: str
    desired_relative_pose: Pose
    translation_tolerance_mm: tuple[float,float,float]
    orientation_tolerance_deg: float
    enabled: bool = True
    orientation_mode: str = 'full'
    yaw_symmetry_deg: float | None = None
    near_scale: float = 2.
    ready_exit_scale: float = 1.1
    safety: SafetyConstraints = field(default_factory=SafetyConstraints)
    axis_labels: tuple[str,str,str] = ('Axis 1','Axis 2','Height')
    tolerance_status: str = 'PROVISIONAL_UI_TOLERANCE'
    # Simulator adapters use these optional descriptors; pure evaluator does not.
    tool_site_name: str = 'fr3_gripper_center_site'
    finger_body_names: tuple[str,...] = ('fr3_left_finger','fr3_right_finger')
    robot_root_body_name: str = 'base'
    table_geom_name: str | None = None
    source: str = ''

    def __post_init__(self):
        if len(self.translation_tolerance_mm)!=3 or any(not math.isfinite(v) or v<=0 for v in self.translation_tolerance_mm):
            raise ValueError('Translation tolerances must be positive')
        if not math.isfinite(self.orientation_tolerance_deg) or self.orientation_tolerance_deg<=0:
            raise ValueError('Orientation tolerance must be positive')
        if self.orientation_mode not in ('yaw','full') or not all(math.isfinite(v) for v in (self.ready_exit_scale,self.near_scale)) or not 1<=self.ready_exit_scale<=self.near_scale:
            raise ValueError('Invalid readiness envelope')
        if len(self.axis_labels) != 3:
            raise ValueError('Three local axis labels required')
        if any(not math.isfinite(v) or v < 0 for _,v in self.safety.minimum_clearance_mm):
            raise ValueError('Safety clearance minima must be finite and nonnegative')
        if self.yaw_symmetry_deg is not None:
            if not math.isfinite(self.yaw_symmetry_deg):raise ValueError('Symmetry must be finite')
            n=360/self.yaw_symmetry_deg if self.yaw_symmetry_deg>0 else 0
            if n<1 or abs(n-round(n))>1e-8:raise ValueError('Symmetry must divide 360 degrees')


@dataclass(frozen=True)
class SafetyReadings:
    clearance_mm: Mapping[str,float | None] = field(default_factory=dict)
    contacts: Mapping[str,bool | None] = field(default_factory=dict)


def orientation_error(current: np.ndarray, desired: np.ndarray,
                      symmetry_deg: float | None = None) -> tuple[float,float]:
    """Shortest full angle and signed reference-Z yaw, with declared axial symmetry."""
    angles=[0.] if symmetry_deg is None else np.arange(round(360/symmetry_deg))*symmetry_deg
    candidates=[]
    for degrees in angles:
        a=np.radians(degrees);c,s=np.cos(a),np.sin(a)
        equivalent=np.array([[c,-s,0],[s,c,0],[0,0,1]])@desired
        delta=current@equivalent.T
        full=float(np.degrees(np.arccos(np.clip((np.trace(delta)-1)/2,-1,1))))
        yaw=float(np.degrees(np.arctan2(delta[1,0],delta[0,0])))
        candidates.append((full,yaw))
    return min(candidates,key=lambda v:v[0])


def unavailable(reason: str = 'No grasp target configured') -> dict:
    return dict(available=False,state='UNAVAILABLE',label='UNAVAILABLE',reason=reason,corrections=[])


def evaluate(config: GraspGuidanceConfig | None, reference: Pose | None,
             current_tool: Pose | None, safety: SafetyReadings | None = None,
             previous_state: str | None = None) -> dict:
    if config is None or not config.enabled:return unavailable()
    if reference is None or current_tool is None:return unavailable('Grasp reference unavailable')
    try:
        ref_rotation=reference.rotation();desired_rotation=config.desired_relative_pose.rotation()
        reference_position=np.asarray(reference.position,dtype=float);current_position=np.asarray(current_tool.position,dtype=float)
        desired_local=np.asarray(config.desired_relative_pose.position,dtype=float)
        if not all(v.shape==(3,) and np.isfinite(v).all() for v in [reference_position,current_position,desired_local]):
            raise ValueError('Invalid position')
        error=(ref_rotation.T@(current_position-reference_position)-desired_local)*1000
        full,yaw=orientation_error(ref_rotation.T@current_tool.rotation(),desired_rotation,config.yaw_symmetry_deg)
    except (ValueError,TypeError):return unavailable('Invalid grasp pose')
    safety=safety or SafetyReadings();failures=[]
    for key,minimum in config.safety.minimum_clearance_mm:
        value=safety.clearance_mm.get(key)
        if value is None or not math.isfinite(value):failures.append('Clearance unavailable')
        elif value<minimum:failures.append('Insufficient clearance')
    for name in config.safety.forbidden_contacts:
        value=safety.contacts.get(name)
        if value is None:failures.append('Contact safety unavailable')
        elif value:failures.append('Forbidden contact: '+name.replace('_',' '))
    angle=abs(yaw) if config.orientation_mode=='yaw' else full
    ratios=np.abs(error)/np.asarray(config.translation_tolerance_mm)
    maximum=max(float(ratios.max()),angle/config.orientation_tolerance_deg)
    ready_scale=config.ready_exit_scale if previous_state=='READY' else 1.
    state='NOT_READY' if failures else 'READY' if maximum<=ready_scale+1e-9 else 'NEAR' if maximum<=config.near_scale else 'NOT_READY'
    corrections=[]
    for i,(label,value,ratio) in enumerate(zip(config.axis_labels,error,ratios)):
        direction=('↓' if value>0 else '↑') if i==2 else ('− '+label if value>0 else '+ '+label)
        corrections.append(dict(label=label,error=float(value),unit='mm',direction=direction,priority=float(ratio)))
    corrections.append(dict(label='Yaw',error=yaw,unit='deg',direction='↻' if yaw>0 else '↺',priority=abs(yaw)/config.orientation_tolerance_deg))
    if config.orientation_mode=='full' and full>max(abs(yaw)*1.2,.05):
        corrections.append(dict(label='Tool angle',error=full,unit='deg',direction='Align tilt',priority=full/config.orientation_tolerance_deg))
    corrections=sorted((v for v in corrections if v['priority']>.01),key=lambda x:x['priority'],reverse=True)[:3]
    return dict(available=True,state=state,label={'READY':'GRIP NOW','NEAR':'ALIGN','NOT_READY':'NOT READY'}[state],
                translation_error_local_mm=error.tolist(),yaw_error_deg=yaw,orientation_error_deg=full,
                clearance_mm={k:v if v is not None and math.isfinite(v) else None for k,v in safety.clearance_mm.items()},safe=not failures,reason=failures[0] if failures else '',
                corrections=corrections,tolerance_status=config.tolerance_status,reference_name=config.reference_name,
                target_position_world=(reference_position+ref_rotation@desired_local).tolist(),advisory_only=True)
