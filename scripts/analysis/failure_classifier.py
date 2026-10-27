"""Restore the original XY20 phase predicates; only sample count is generalized."""
import numpy as np

RULES=dict(ee_motion_threshold_m=.02,joint_motion_threshold_rad=.15,approach_distance_reduction_m=.02,
    approach_proximity_m=.1,acquired_rise_from_reset_m=.02,acquired_ee_object_proximity_m=.12,
    acquired_consecutive_frames=5,transport_xy_displacement_m=.1,transport_goal_distance_reduction_m=.1,
    insertion_xy_axis_tolerance_m=.05,insertion_descent_from_previous_peak_m=.02,insertion_height_above_goal_m=.1,
    acquisition_interpretation='sustained lift with EE proximity is a kinematic acquisition proxy; finger-object contacts are not individually logged',
    approach_interpretation='EE/object geometry plus EE motion; object XY below 2 cm does not imply a stationary robot',
    classification='furthest achieved observable stage A/B/C/D/E/F; does not alter termination or task success')
STAGE_NAMES=dict(A='no_meaningful_approach',B='approach_without_acquisition',C='acquired_lifted_no_transport',
    D='transport_without_insertion_attempt',E='insertion_attempt_without_success',F='task_success')
def sustained(mask,n):
    return bool(len(mask)>=n and (np.convolve(mask.astype(int),np.ones(n,dtype=int),mode='valid')>=n).any())
def phase(result,state,obj,ee,q):
    initial=np.asarray(result['initial_object_position']);goal=np.asarray(result['initial_goal_position'])
    np.testing.assert_array_equal(initial[:2],[state['x'],state['y']])
    max_xy=float(np.linalg.norm(obj[:,:2]-initial[:2],axis=1).max())
    ee_move=float(np.linalg.norm(ee-ee[0],axis=1).max());joint_move=float(np.abs(q-q[0]).max())
    ee_obj=np.linalg.norm(ee-obj,axis=1)
    approach_reduction=float(ee_obj[0]-ee_obj.min())
    motion=ee_move>=RULES['ee_motion_threshold_m'] or joint_move>=RULES['joint_motion_threshold_rad']
    approach=motion and (approach_reduction>=RULES['approach_distance_reduction_m'] or
        (ee_move>=RULES['ee_motion_threshold_m'] and ee_obj.min()<=RULES['approach_proximity_m']))
    acquire_mask=(obj[:,2]>=initial[2]+RULES['acquired_rise_from_reset_m']) & (ee_obj<=RULES['acquired_ee_object_proximity_m'])
    acquired=sustained(acquire_mask,RULES['acquired_consecutive_frames'])
    acquisition_index=None
    if acquired:
        n=RULES['acquired_consecutive_frames']
        acquisition_index=int(np.flatnonzero(np.convolve(acquire_mask.astype(int),np.ones(n,dtype=int),mode='valid')>=n)[0]+n-1)
    after_acquisition=np.arange(len(obj))>=acquisition_index if acquired else np.zeros(len(obj),dtype=bool)
    goal_distance=np.linalg.norm(obj[:,:2]-goal[:2],axis=1)
    transport_mask=after_acquisition & (np.linalg.norm(obj[:,:2]-initial[:2],axis=1)>=RULES['transport_xy_displacement_m']) & (np.linalg.norm(initial[:2]-goal[:2])-goal_distance>=RULES['transport_goal_distance_reduction_m'])
    transport=bool(transport_mask.any())
    near_peg=np.all(np.abs(obj[:,:2]-goal[:2])<=RULES['insertion_xy_axis_tolerance_m'],axis=1)
    descent=np.maximum.accumulate(obj[:,2])-obj[:,2]
    insertion=acquired and bool((after_acquisition & near_peg & (descent>=RULES['insertion_descent_from_previous_peak_m']) &
        (obj[:,2]<=goal[2]+RULES['insertion_height_above_goal_m'])).any())
    success=result['result']=='SUCCESS'
    stage='F' if success else 'E' if insertion else 'D' if transport else 'C' if acquired else 'B' if approach else 'A'
    return dict(failure_stage=stage,failure_stage_label=STAGE_NAMES[stage],acquisition_proxy=acquired,
        approach=approach,transport=transport,insertion_attempt=insertion,robot_motion=motion,
        first_acquired_policy_step=acquisition_index+1 if acquisition_index is not None else None,
        first_transport_policy_step=int(np.flatnonzero(transport_mask)[0])+1 if transport else None,
        max_object_xy_displacement=max_xy,max_object_z=float(obj[:,2].max()),
        object_rise_from_reset=float(obj[:,2].max()-initial[2]),max_ee_displacement=ee_move,
        max_arm_joint_change=joint_move,min_ee_object_distance=float(ee_obj.min()),
        ee_object_distance_reduction=approach_reduction,min_object_goal_xy_distance=float(goal_distance.min()))
