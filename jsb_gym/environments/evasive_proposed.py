# evasive_proposed.py — shaping_w (보상 가중치) 인자 추가

from jsb_gym.TAU.aircraft import F16
from jsb_gym.TAU.missiles import AIM
from jsb_gym.utils.utils import toolkit, Geo
import numpy as np
from geopy.distance import geodesic

class Evasive(object):
    def __init__(self, conf, args , aim_evs, f16_evs):
        self.conf = conf
        self.tk = toolkit()
        self.gtk = Geo()
        self.prev_actions = []
        self.action_history = []
        self.shaping_w = args.get("shaping_w", 0.1)  # 🧠 default = 0.1
        if self.conf.general['rec_f16']:
            from jsb_gym.utils.tb_logs import F16_logs
            f16_logs = F16_logs(conf)
            self.f16 = F16(conf = f16_evs, FlightGear= args['vizualize'], logs= f16_logs)
        else:
            self.f16 = F16(conf = f16_evs, FlightGear= args['vizualize'])

        self.aim_block_names = list(self.conf.aim.keys())
        self.aim_block = {k: AIM(aim_evs, args['vizualize'], None) for k in self.aim_block_names}
        self.states_extra = {k: None for k in self.aim_block_names}

        self.r_step = range(self.conf.general['fg_r_step'] if args['vizualize'] else self.conf.general['r_step'])
        self.env_name = self.conf.general['env_name']
        self.f16.name  = self.conf.general['f16_name']
        self.sim_time_sec_max = self.conf.general['sim_time_max']

        self.observation_space = self.conf.states['obs_space']
        for i in self.aim_block_names:
            self.f16.state_block[i] = np.empty((1,self.conf.states['obs_space']))
        self.action_space = np.empty((1,self.conf.states['act_space']))

    def update_states(self):
        for key in self.aim_block:
            self.f16.state_block[key][0,0] = self.get_distance_to_firing_position(self.f16, self.aim_block[key], scale=True)

    def get_distance_to_firing_position(self, f16, aim, scale=False, offset=None):
        f16 = (f16.get_lat_gc_deg(), f16.get_long_gc_deg())
        aim = (aim.lat0, aim.long0)
        dist = geodesic(aim, f16).meters
        if offset is not None:
            dist += offset
        return self.tk.scale_between(dist, self.conf.sf['d_min'], self.conf.sf['d_max']) if scale else dist

    def get_closing_velocity(self, f16, aim):
        rel_vel = np.array(aim.vel0_vec) - np.array([
            f16.get_v_north(), f16.get_v_east(), f16.get_v_down()
        ])
        rel_pos = np.array(self.get_relative_unit_position_NED(f16, aim))
        rel_dist = np.linalg.norm(rel_pos)
        if rel_dist == 0:
            return 0
        return np.dot(rel_vel, rel_pos) / rel_dist

    def get_relative_unit_position_NED(self, f16, aim):
        lat0 = f16.get_lat_gc_deg()
        lon0 = f16.get_long_gc_deg()
        h0   = f16.get_altitude()
        lat, lon, h = aim.lat0, aim.long0, aim.alt0
        east, north, down = self.gtk.get_relative_unit_position_NED(lat0, lon0, h0, lat, lon, h)
        return north, east, down

    def is_done(self):
        for key in self.aim_block:
            if self.aim_block[key].is_target_hit():
                self.f16_alive = False
                return True
            if self.aim_block[key].is_traget_lost():
                return True
        if self.f16.get_altitude() < 1e3 or self.f16.get_sim_time_sec() > self.sim_time_sec_max:
            return True
        return False

    def get_reward(self, is_done):
        reward = 0.01
        for key in self.aim_block:
            dist = self.get_distance_to_firing_position(self.f16, self.aim_block[key])
            closing_vel = self.get_closing_velocity(self.f16, self.aim_block[key])
            threat = 0
            if dist < 15000: threat += 0.5
            if closing_vel > 100: threat += 0.5
            reward -= self.shaping_w * threat  # 🧠 shaping 가중치 반영

        if len(self.prev_actions) > 1:
            delta = np.linalg.norm(np.array(self.prev_actions[-1]) - np.array(self.prev_actions[-2]))
            reward += 0.05 * delta

        if is_done:
            reward += 1.0 if self.f16_alive else -1.0

        return reward

    def step(self, action, action_type):
        self.prev_actions.append(action)
        self.action_history.append(action.tolist())
        for _ in self.r_step:
            self.f16.step_BVR(action, action_type=action_type)
            for key in self.aim_block:
                if not self.aim_block[key].is_traget_lost():
                    self.aim_block[key].step_evasive()
        done = self.is_done()
        reward = self.get_reward(done)
        self.update_states()
        return self.f16.state_block, reward, done, None

    def reset(self, rand_state_f16=False, rand_state_aim=False):
        self.f16_alive = True
        self.prev_actions = []
        self.action_history = []
        lat0, long0, alt0, vel0, heading0 = 37.0, 127.0, 10000, 250, 0.0
        self.f16.reset(lat0, long0, alt0, vel0, heading0)
        for key in self.aim_block:
            lat, long, alt, vel, heading = 37.05, 127.05, 11000, 300, 180.0
            self.aim_block[key].reset(lat, long, alt, vel, heading)
            self.aim_block[key].reset_target(self.f16, set_active=True)
        self.update_states()
        return self.f16.state_block
