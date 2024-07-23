import json
import habitat
from habitat import get_config
from habitat.sims import make_sim
from habitat.utils.visualizations import maps

import numpy as np

from semantic_utils import use_fine, object_whitelist
from semantic_utils import replica_to_mp3d_12cat_mapping




class HabitatUtils:
    def __init__(self, scene, level, hfov, h, w, housetype='mp3d'):
        # -- scene = data/mp3d/house/house.glb
        self.scene = scene
        self.level = level  # -- int
        self.house = scene.split('/')[-2]
        self.housetype = housetype

        #-- setup config
        self.config = get_config()
        self.config.defrost()
        self.config.SIMULATOR.SCENE = scene
        self.config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR",
                                                 "DEPTH_SENSOR",
                                                 "SEMANTIC_SENSOR"]
        self.config.SIMULATOR.RGB_SENSOR.HFOV = hfov
        self.config.SIMULATOR.RGB_SENSOR.HEIGHT = h
        self.config.SIMULATOR.RGB_SENSOR.WIDTH = w
        self.config.SIMULATOR.DEPTH_SENSOR.HFOV = hfov
        self.config.SIMULATOR.DEPTH_SENSOR.HEIGHT = h
        self.config.SIMULATOR.DEPTH_SENSOR.WIDTH = w
        self.config.SIMULATOR.SEMANTIC_SENSOR.HFOV = hfov
        self.config.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = h
        self.config.SIMULATOR.SEMANTIC_SENSOR.WIDTH = w
        # self.config.SIMULATOR.AGENT_0.HEIGHT = 0

        # -- Original resolution
        self.config.SIMULATOR.FORWARD_STEP_SIZE = 0.1
        self.config.SIMULATOR.TURN_ANGLE = 9

        # -- fine resolution setps
        #self.config.SIMULATOR.FORWARD_STEP_SIZE = 0.05
        #self.config.SIMULATOR.TURN_ANGLE = 3

        # -- render High Rez images
        #self.config.SIMULATOR.RGB_SENSOR.HEIGHT = 720
        #self.config.SIMULATOR.RGB_SENSOR.WIDTH = 1280

        # -- LOOK DOWN
        #theta = 30 * np.pi / 180
        #self.config.SIMULATOR.RGB_SENSOR.ORIENTATION = [-theta, 0.0, 0.0]
        #self.config.SIMULATOR.DEPTH_SENSOR.ORIENTATION = [-theta, 0.0, 0.0]
        #self.config.SIMULATOR.SEMANTIC_SENSOR.ORIENTATION = [-theta, 0.0, 0.0]

        # -- OUTDATED (might be able to re-instantiate those in future commits)
        #self.config.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD",
        #                                     "TURN_LEFT", "TURN_RIGHT",
        #                                     "LOOK_UP", "LOOK_DOWN"]

        # -- ObjNav settings
        #self.config.SIMULATOR.FORWARD_STEP_SIZE = 0.25
        #self.config.SIMULATOR.TURN_ANGLE = 30


        self.config.freeze()

        self.agent_height = self.config.SIMULATOR.AGENT_0.HEIGHT

        self.sim = make_sim(id_sim=self.config.SIMULATOR.TYPE, config=self.config.SIMULATOR)

        self.semantic_annotations = self.sim.semantic_annotations()

        self.sim.reset()

        agent_state = self.get_agent_state()
        self.position = agent_state.position
        self.rotation = agent_state.rotation

        # -- get level dimensions
        # -- read it directly from the saved data from the .house files
        # Tries to set the agent on the given floor. It's actually quite hard..
        # if housetype == 'mp3d':
        #     env = '_'.join([self.house, str(self.level)])
        #     houses_dim = json.load(open('data/houses_dim.json', 'r'))
        #     self.center = np.array(houses_dim[env]['center'])
        #     self.sizes = np.array(houses_dim[env]['sizes'])
        #     self.start_height = self.center[1] - self.sizes[1]/2

        #     self.set_agent_on_level()
        # else:
        #     pass

        self.all_objects = self.get_objects_in_house()

    @property
    def position(self):
        return self._position


    @position.setter
    def position(self, p):
        self._position = p


    @property
    def rotation(self):
        return self._rotation


    @rotation.setter
    def rotation(self, r):
        self._rotation = r


    def set_agent_state(self):
        self.sim.set_agent_state(self._position,
                                 self._rotation)

    def get_agent_state(self):
        return self.sim.get_agent_state()


    def get_sensor_pos(self):
        ags = self.sim.get_agent_state()
        return ags.sensor_states['rgb'].position

    def get_sensor_ori(self):
        ags = self.sim.get_agent_state()
        return ags.sensor_states['rgb'].rotation



    def reset(self):
        self.sim.reset()
        agent_state = self.get_agent_state()
        self.position = agent_state.position
        self.rotation = agent_state.rotation

    def set_agent_on_level(self):
        """
        It is very hard to know exactly the z value of a level as levels can
        have stairs and difference in elevation etc..
        We use the level.aabb to get an idea of the z-value of the level but
        that isn't very robust (eg r1Q1Z4BcV1o_0 -> start_height of floor 0:
            -1.3 but all sampled point will have a z-value around 0.07, when
            manually naivagting in the env we can see a pth going downstairs..)
        """
        point = self.sample_navigable_point()
        self.position = point
        self.set_agent_state()

    def step(self, action):
        self.sim.step(action)


    def sample_navigable_point(self):
        """
        If house has only one level we sample directly a nav point
        Else we iter until we get a point on the right floor..
        """
        if len(self.semantic_annotations.levels) == 1:
            return self.sim.sample_navigable_point()
        else:
            for _ in range(1000):
                point = self.sim.sample_navigable_point()
                #return point
                if np.abs(self.start_height - point[1]) <= 1.5:
                #if np.all(((self.center-self.sizes/2)<=point) &
                #          ((self.center+self.sizes/2)>=point)):
                    return point
            print('No navigable point on this floor')
            return None


    def sample_rotation(self):
        theta = np.random.uniform(high=np.pi)
        quat = np.array([0, np.cos(theta/2), 0, np.sin(theta/2)])
        return quat



    def get_house_dimensions(self):
        return self.semantic_annotations.aabb



    def get_objects_in_scene(self):
        """

            returns dict with {int obj_id: #pixels in frame}

        """
        buf = self.sim.render(mode="semantic")
        unique, counts = np.unique(buf, return_counts=True)
        objects = {int(u): c for u, c in zip(unique, counts)}
        return objects


    def render(self, mode='rgb'):
            return self.sim.render(mode=mode)



    def render_semantic_mpcat40(self):
        buf = self.sim.render(mode="semantic")
        out = np.zeros(buf.shape, dtype=np.uint8) # class 0 -> void
        object_ids = np.unique(buf)
        for oid in object_ids:
            object = self.all_objects[oid]
            # -- mpcat40_name = object.category.name(mapping='mpcat40')
            mpcat40_index = object.category.index(mapping='mpcat40')
            # remap everything void/unlabeled/misc/etc .. to misc class 40
            # (void:0,  unlabeled: 41, misc=40)
            if mpcat40_index <= 0 or mpcat40_index > 40: mpcat40_index = 40 # remap -1 to misc
            out[buf==oid] = mpcat40_index
        return out



    def render_semantic_12cat(self):
        buf = self.sim.render(mode="semantic")
        out = np.zeros(buf.shape, dtype=np.uint8) # class 0 -> void
        object_ids = np.unique(buf)
        for oid in object_ids:
            object = self.all_objects[oid]
            object_name = object.category.name(mapping='mpcat40')
            if object_name in use_fine:
                object_name = object.category.name(mapping='raw')
            if object_name in object_whitelist:
                object_index = object_whitelist.index(object_name)+1
                out[buf==oid] = object_index
        return out

    def render_semantic_12cat_replica(self):
        buf = self.sim.render(mode="semantic")
        out = np.zeros(buf.shape, dtype=np.uint8) # class 0 -> void
        object_ids = np.unique(buf)
        for oid in object_ids:
            if oid in self.all_objects:
                object = self.all_objects[oid]
                if object.category is not None:
                    object_name = object.category.name()
                    if object_name in replica_to_mp3d_12cat_mapping:
                        object_name = replica_to_mp3d_12cat_mapping[object_name]
                        object_index = object_whitelist.index(object_name)+1
                        out[buf==oid] = object_index
        return out



    def get_objects_in_level(self):
        # /!\ /!\ level IDs are noisy in MP3D
        # /!\ /!\

        if self.housetype == 'mp3d':

            assert self.level == int(self.semantic_annotations.levels[self.level].id)

            objects = {}
            for region in self.semantic_annotations.levels[self.level].regions:
                for object in region.objects:
                    objects[int(object.id.split('_')[-1])] = object
        else:
            objects = self.all_objects

        return objects


    def get_objects_in_house(self):
        objects = {int(o.id.split('_')[-1]): o for o in self.semantic_annotations.objects if o is not None}
        return objects


    def get_topdown_map(self, map_resolution=(1250,1250), num_samples=20000, draw_border=True):

        assert np.abs(self.start_height - self.position[1]) <= 1.5

        house_aabb = self.get_house_dimensions()

        min_x = house_aabb.center[0] - house_aabb.sizes[0]/2
        max_x = house_aabb.center[0] + house_aabb.sizes[0]/2
        min_z = house_aabb.center[2] - house_aabb.sizes[2]/2
        max_z = house_aabb.center[2] + house_aabb.sizes[2]/2

        min_coord = min(min_x, min_z)
        max_coord = max(max_x, max_z)

        maps.COORDINATE_MIN = min_coord
        maps.COORDINATE_MAX = max_coord

        map = maps.get_topdown_map(self.sim,
                                   map_resolution=map_resolution,
                                   num_samples=num_samples,
                                   draw_border=draw_border)

        return map


    def draw_agent(self, map, agent_center_coord, agent_rotation, agent_radius_px=5):
        maps.draw_agent(map, agent_center_coord, agent_rotation, agent_radius_px=agent_radius_px)
        return map


    def keep_objects_in_whitelist(self, objects):
        kept_objects = {}
        for oid, o in objects.items():
            name = o.category.name(mapping='mpcat40')
            if name in use_fine:
                name = o.category.name(mapping='raw')
            if name in object_whitelist:
                kept_objects[oid] = o
        return kept_objects

    def __del__(self):
        self.sim.close()
