"""MuJoCo pose and collision-mesh safety adapter for generic grasp guidance."""
from __future__ import annotations

import mujoco
import numpy as np
from .grasp_guidance import Pose, SafetyReadings, evaluate, unavailable


def pose(position,rotation):
    q=np.zeros(4);mujoco.mju_mat2Quat(q,np.asarray(rotation).reshape(9))
    return Pose(tuple(position),tuple(q))


class MujocoGraspProvider:
    def __init__(self,model,config):
        self.model=model;self.config=config;self.previous_state=None;self.problem=None
        if config is None or not config.enabled:return
        try:
            lookup={'site':model.site,'body':model.body,'geom':model.geom}
            self.reference_id=lookup[config.reference_type](config.reference_name).id
            self.tool_id=model.site(config.tool_site_name).id
            root=model.body(config.robot_root_body_name).id
            def descendant(body,ancestor):
                while body>0:
                    if body==ancestor:return True
                    body=int(model.body_parentid[body])
                return False
            self.robot_bodies={i for i in range(model.nbody) if descendant(i,root)}
            finger_roots=[model.body(name).id for name in config.finger_body_names]
            self.finger_bodies={i for i in range(model.nbody) if any(descendant(i,b) for b in finger_roots)}
            self.vertices={}
            for i in range(model.ngeom):
                if model.geom_bodyid[i] not in self.finger_bodies or not (model.geom_contype[i] or model.geom_conaffinity[i]):continue
                if model.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH:
                    mesh=int(model.geom_dataid[i]);start=int(model.mesh_vertadr[mesh]);n=int(model.mesh_vertnum[mesh])
                    self.vertices[i]=model.mesh_vert[start:start+n].astype(float).copy()
                elif model.geom_type[i]==mujoco.mjtGeom.mjGEOM_BOX:
                    self.vertices[i]=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*model.geom_size[i]
            self.table_id=model.geom(config.table_geom_name).id if config.table_geom_name else None
            self.table_body=int(model.geom_bodyid[self.table_id]) if self.table_id is not None else None
        except (KeyError,ValueError) as exc:self.problem=str(exc)

    def sample(self,data):
        config=self.config
        if config is None or not config.enabled:return unavailable()
        if self.problem:return unavailable('Configured grasp reference unavailable')
        kind=config.reference_type;i=self.reference_id
        reference=pose(getattr(data,{'site':'site_xpos','body':'xpos','geom':'geom_xpos'}[kind])[i],
                       getattr(data,{'site':'site_xmat','body':'xmat','geom':'geom_xmat'}[kind])[i])
        current=pose(data.site_xpos[self.tool_id],data.site_xmat[self.tool_id])
        clearance={'finger_table':None,'ee_table':None};contacts={}
        if self.table_id is not None and self.model.geom_type[self.table_id]==mujoco.mjtGeom.mjGEOM_BOX:
            table_position=data.geom_xpos[self.table_id];table_rotation=data.geom_xmat[self.table_id].reshape(3,3)
            # Signed minimum vertex gap to the table's top plane. Collision mesh
            # only; conservative plane extension, not camera-pixel clearance.
            top=float(self.model.geom_size[self.table_id,2])
            values=[]
            for geom,vertices in self.vertices.items():
                # Only the plane-normal component is needed. Avoid building
                # two N-by-3 transformed meshes every display update.
                normal=table_rotation[:,2]
                axis=data.geom_xmat[geom].reshape(3,3).T@normal
                heights=vertices@axis+(data.geom_xpos[geom]-table_position)@normal
                values.append(float(heights.min()-top)*1000)
            if values:clearance['finger_table']=min(values)
            clearance['ee_table']=float(((data.site_xpos[self.tool_id]-table_position)@table_rotation)[2]-top)*1000
            contacts={'finger_table':False,'robot_table':False}
            for c in data.contact:
                if c.efc_address<0:continue
                a,b=int(self.model.geom_bodyid[c.geom1]),int(self.model.geom_bodyid[c.geom2])
                if self.table_body not in (a,b):continue
                other=b if a==self.table_body else a
                contacts['finger_table']|=other in self.finger_bodies
                contacts['robot_table']|=other in self.robot_bodies
        result=evaluate(config,reference,current,SafetyReadings(clearance,contacts),self.previous_state)
        self.previous_state=result['state']
        result['sim_time']=float(data.time)
        return result
