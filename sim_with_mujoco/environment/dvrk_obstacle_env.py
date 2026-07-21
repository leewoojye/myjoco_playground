import mujoco

from sim_with_mujoco.environment.env import DvrkEnv


class DvrkNeedleObstacleEnv(DvrkEnv):
    TIP_RADIUS = 0.006

    def __init__(self, xml_path, **kwargs):
        super().__init__(xml_path, **kwargs)
        geom_id = self.get_id(mujoco.mjtObj.mjOBJ_GEOM, "needle_path_obstacle")
        site_id = self.get_id(mujoco.mjtObj.mjOBJ_SITE, "needle_path_obstacle_center")
        self.obstacle_radius = float(self.model.geom_size[geom_id, 0])
        self.obstacle_position = self.data.site_xpos[site_id].copy()
