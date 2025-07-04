from jsb_gym.TAU.aircraft import F16
from jsb_gym.TAU.missiles import AIM
from jsb_gym.utils.tb_logs import Env_logs
from jsb_gym.utils.utils import toolkit, Geo
import numpy as np
from geopy.distance import geodesic

# log files if needed 
from jsb_gym.utils.tb_logs import F16_logs
from jsb_gym.utils.tb_logs import AIM_logs


class Evasive(object):
    def __init__(self, conf, args , aim_evs, f16_evs):
        # Evasive config file 
        self.conf = conf
        # transformation/scalling tools 
        self.tk = toolkit()
        self.gtk = Geo()
        # Logs for the environment 
        # usually used for all units wihtin the environment  

        #self.reset_logs()
        #High frequency recodring 
        if self.conf.general['rec_f16']:
            # record flight data of F16
            f16_logs = F16_logs(conf)
            self.f16 = F16(conf = f16_evs, FlightGear= args['vizualize'], logs= f16_logs)
        else: 
            # scip recoring flight data 
            self.f16 = F16(conf = f16_evs, FlightGear= args['vizualize'])

        self.states_extra = {}
        
        self.aim_block_names = list(self.conf.aim.keys())
        self.aim_block = {}            
        for i in self.aim_block_names:
            if len(self.conf.aim) == 1:
                fg = args['vizualize']
                fg_out = 'data_output/flightgear_red.xml'
                if self.conf.general['rec_aim']:
                    aim_logs = AIM_logs(conf)
                else:
                    aim_logs = None
            else:
                fg = False
                fg_out = None
                aim_logs = None

            self.aim_block[i] = AIM(aim_evs, fg, fg_out, logs= aim_logs)
            # store location of the firing position 
            self.states_extra[i] = None

        if args['vizualize']:
            self.r_step = range(self.conf.general['fg_r_step'])
        else:
            self.r_step = range(self.conf.general['r_step'])

        # general configuration 
        self.env_name = self.conf.general['env_name']
        self.f16.name  = self.conf.general['f16_name']
        self.sim_time_sec_max = self.conf.general['sim_time_max']

        # load state holder
        self.observation_space = self.conf.states['obs_space']
        for i in self.aim_block_names:
            self.f16.state_block[i] = np.empty((1,self.conf.states['obs_space']))
 
        self.action_space = np.empty((1,self.conf.states['act_space']))
        self.trajectory_log = None # 궤적 로그를 담을 변수


    def get_init_state_F16(self, rand_state = False):
        if rand_state:
            lat  = np.random.uniform(self.conf.f16_rand['lat'][0], self.conf.f16_rand['lat'][1])
            long = np.random.uniform(self.conf.f16_rand['long'][0], self.conf.f16_rand['long'][1])
            vel  = np.random.uniform(self.conf.f16_rand['vel'][0], self.conf.f16_rand['vel'][1])
            alt  = np.random.uniform(self.conf.f16_rand['alt'][0], self.conf.f16_rand['alt'][1])
            head = np.random.uniform(self.conf.f16_rand['heading'][0], self.conf.f16_rand['heading'][1])       
        
        else:
            lat = self.conf.f16['lat']
            long = self.conf.f16['long']
            alt = self.conf.f16['alt']
            vel = self.conf.f16['vel']
            head = self.conf.f16['heading']
        
        return lat, long, alt, vel, head

    def get_init_state_AIM(self, name, lat_tgt, long_tgt, rand_state = False):
        # 
        if rand_state:
            lat, long, d, b = self.gtk.get_random_position_in_circle(lat0=lat_tgt,
                                                   long0= long_tgt,
                                                   d= self.conf.aim_rand[name]['distance'],
                                                   b = self.conf.aim_rand[name]['bearing'])
            
            alt = np.random.uniform(self.conf.aim_rand[name]['alt'][0], 
                                    self.conf.aim_rand[name]['alt'][1])
            
            vel = np.random.uniform(self.conf.aim_rand[name]['vel'][0], 
                                    self.conf.aim_rand[name]['vel'][1])
            
            heading = self.gtk.get_bearing(lat, long, lat_tgt, long_tgt)

        else:

            lat, long = self.gtk.db2latlong(lat0= lat_tgt, long0=long_tgt,
                                d= self.conf.aim[name]['distance'], 
                                b= self.conf.aim[name]['bearing'])
            alt = self.conf.aim[name]['alt']
            vel = self.conf.aim[name]['vel']

            heading = self.gtk.get_bearing(lat, long, lat_tgt, long_tgt)
            
        return lat, long, alt, vel, heading 

    def get_distance_to_firing_position(self, f16, aim, scale=False, offset = None):
        
        f16 = (f16.get_lat_gc_deg(), f16.get_long_gc_deg())
        aim = (aim.lat0, aim.long0)
        dist = geodesic(aim, f16).meters
        
        if offset != None:
            dist += offset

        # scale between -1 to 1
        if scale:
            dist = self.tk.scale_between(a= dist,\
                                         a_min= self.conf.sf['d_min'],\
                                         a_max=self.conf.sf['d_max'])
        return dist 
    
    def get_time_since_missile_active(self, f16, scale = False, offset = None):
        flight_time = f16.get_sim_time_sec()

        if offset != None:
            flight_time += offset
            if flight_time < 0:
                flight_time = 0

        if scale:
            flight_time = self.tk.scale_between(a=flight_time, a_min= 0,\
                                                a_max = self.conf.sf['t'])
        return flight_time 
    
    def get_angle_to_firing_position(self, f16, aim, scale= False, offset = None):

        aim_lat = aim.lat0
        aim_long = aim.long0
        f16_lat = f16.get_lat_gc_deg()
        f16_long = f16.get_long_gc_deg()

        ref_yaw = self.gtk.get_bearing(f16_lat, f16_long, aim_lat, aim_long)
        self.affp = ref_yaw
        if offset != None:
            ref_yaw += offset
        
        # range -180 to 180 deg 
        ref_yaw = self.tk.get_heading_difference(psi_ref= ref_yaw, psi_deg= self.f16.fdm['attitude/psi-deg'])
        
        if scale:
            # eliminate breakpoint 
            return np.sin(np.radians(ref_yaw)), np.cos(np.radians(ref_yaw))
        else:
            return ref_yaw 

    def get_velocity(self, scale=False):
        vel = self.f16.get_Mach()
        if scale:
            # max speed mach 2
            vel = self.tk.scale_between(a= vel, a_min= 0, a_max= self.conf.sf['mach_max'])
            return vel
        
        else:
            return vel 

    def get_altitude(self, scale = False):
        alt = self.f16.get_altitude()
        if scale:
            alt = self.tk.scale_between(a= alt, a_min=self.conf.sf['alt_min'], a_max= self.conf.sf['alt_max'] )
        return alt 

    def get_theta(self, scale = False):
        if scale:
            return self.f16.get_theta(scaled= True)
        else:
            return self.f16.get_theta(scaled= False)

    def get_psi(self, scale= False):
        if scale:
            return self.f16.get_psi(scaled= True)
        else:
            return self.f16.get_psi(scaled= False)

    def get_aim_vel0(self, aim, scale = False):
        if scale:
            # max speed mach 2
            vel = self.tk.scale_between(a= aim.vel0, 
                                        a_min= self.conf.sf['aim_vel0_min'], 
                                        a_max= self.conf.sf['aim_vel0_max'])
            return vel

    def get_aim_alt0(self, aim, scale = False):
        if scale:
            # max speed mach 2
            alt = self.tk.scale_between(a= aim.alt0, 
                                        a_min= self.conf.sf['aim_alt0_min'], 
                                        a_max= self.conf.sf['aim_alt0_max'])
            return alt

    def get_relative_unit_position_NED(self, f16, aim, scale = False):
        # utils.geo.get_relative_unit_position_NED
        # TAU lat0, lon0, h0, 
        lat0 = f16.get_lat_gc_deg()
        lon0 = f16.get_long_gc_deg()
        h0   = f16.get_altitude()

        # Object of interest lat, lon, h
        lat = aim.lat0 
        lon = aim.long0 
        h   = aim.alt0
        
        east, north, down = self.gtk.get_relative_unit_position_NED(lat0, lon0, h0, lat, lon, h)
        if scale:
            east = self.tk.scale_between(a= east,\
                                         a_min= self.conf.states['NE_scale'][0],\
                                         a_max= self.conf.states['NE_scale'][1])
            north = self.tk.scale_between(a= north,\
                                         a_min= self.conf.states['NE_scale'][0],\
                                         a_max= self.conf.states['NE_scale'][1])
            down = self.tk.scale_between(a= down,\
                                         a_min= self.conf.states['D_scale'][0],\
                                         a_max= self.conf.states['D_scale'][1])
        return north, east, down

    def get_relative_velocity_NED(self, f16, aim, scale= False):
        
        v_north =   aim.vel0_vec[0] - f16.get_v_north() 
        v_east  =   aim.vel0_vec[1]  - f16.get_v_east() 
        v_down  =   aim.vel0_vec[2]  - f16.get_v_down()
        if scale:
            v_north = self.tk.scale_between(a= v_north,\
                                         a_min= self.conf.states['v_NED_scale'][0],\
                                         a_max= self.conf.states['v_NED_scale'][1])
            v_east = self.tk.scale_between(a= v_east,\
                                         a_min= self.conf.states['v_NED_scale'][0],\
                                         a_max= self.conf.states['v_NED_scale'][1])
            v_down = self.tk.scale_between(a= v_down,\
                                         a_min= self.conf.states['v_NED_scale'][0],\
                                         a_max= self.conf.states['v_NED_scale'][1])
        return v_north, v_east, v_down

    def get_v_down(self, scale= False):
        if scale:
            return self.f16.get_v_down(scaled= True)
        else:
            return self.f16.get_v_down(scaled= False)
        
    def update_states(self):
        
        if self.conf.states['update_states_type'] == 1:
            for key in self.aim_block:
                self.f16.state_block[key][0,0] = self.get_distance_to_firing_position(self.f16, self.aim_block[key], scale=True)
                self.f16.state_block[key][0,1] = self.get_time_since_missile_active(self.f16, scale=True)
                self.f16.state_block[key][0,2], self.f16.state_block[key][0,3] = self.get_angle_to_firing_position(self.f16, self.aim_block[key], scale=True)
                self.f16.state_block[key][0,4] = self.get_velocity(scale = True)
                self.f16.state_block[key][0,5] = self.get_altitude(scale = True)
                self.f16.state_block[key][0,6] = self.get_psi(scale = True)
                self.f16.state_block[key][0,7] = self.get_aim_vel0(aim=self.aim_block[key],scale = True)
                self.f16.state_block[key][0,8] = self.get_aim_alt0(aim=self.aim_block[key],scale = True)
                self.f16.state_block[key][0,9] = self.get_v_down(scale = True) 
                self.states_extra[key] = self.affp
                
        elif self.conf.states['update_states_type'] == 2:
            for key in self.aim_block:
                north, east, down = self.get_relative_unit_position_NED(f16=self.f16, aim=self.aim_block[key], scale=True)
                self.f16.state_block[key][0,0] = north
                self.f16.state_block[key][0,1] = east
                self.f16.state_block[key][0,2] = down
                
                self.f16.state_block[key][0,3] = self.get_time_since_missile_active(self.f16, scale=True)
                self.f16.state_block[key][0,4] = self.get_altitude(scale = True)
                self.f16.state_block[key][0,5] = self.get_aim_alt0(aim=self.aim_block[key],scale = True)        
                
                v_north, v_east, v_down = self.get_relative_velocity_NED(f16=self.f16, aim=self.aim_block[key], scale= True)
                self.f16.state_block[key][0,6] = v_north
                self.f16.state_block[key][0,7] = v_east
                self.f16.state_block[key][0,8] = v_down
                #self.states_extra[key] = self.affp

        elif self.conf.states['update_states_type'] == 3:
            for key in self.aim_block:
                
                # missile position 
                self.f16.state_block[key][0,0] = self.get_distance_to_firing_position(self.f16, self.aim_block[key], scale=True)
                self.f16.state_block[key][0,1], self.f16.state_block[key][0,2] = self.get_angle_to_firing_position(self.f16, self.aim_block[key], scale=True)
                # altitude               
                self.f16.state_block[key][0,3] = self.get_altitude(scale = True)
                # heading 
                self.f16.state_block[key][0,4] = np.sin(np.radians(self.get_psi(scale = False)))
                self.f16.state_block[key][0,5] = np.cos(np.radians(self.get_psi(scale = False)))
                


    def scale_reward(self, reward):
        return reward/self.conf.sf['d_max_reward']

    def scale_reward_inv(self, reward_scaled):
        return reward_scaled*self.conf.sf['d_max_reward']

    def get_reward(self, is_done):
        method = self.conf.general.get("method", "baseline")
        if hasattr(self.conf, 'reward_weights'):
            weights = self.conf.reward_weights
        else:
            weights = {
                'survival': 0.5,
                'tactical': 0.3,
                'entropy': 0.1,
                'shaping': 0.1
    }

        # 💠 엔트로피 (ppo에서 가져온 최신 entropy 값)
        R_entropy = getattr(self, 'last_entropy', 0.0)

        # 💠 Shaping 보상 예시: 하강 속도 유지
        R_shaping = self.f16.get_v_down(scaled=True)

        # 💠 전술 보상 계산: 거리, ATA, AA, 선회율, 시야 점유율
        aim = list(self.aim_block.values())[0]  # 첫 번째 미사일 기준
        R_distance = self.get_distance_to_firing_position(self.f16, aim)
        R_ATA = abs(self.get_angle_to_firing_position(self.f16, aim))
        R_AA = abs(self.get_psi())
        R_turnrate = abs(self.f16.get_theta())

        # 💠 시야 점유율 (가정: AIM이 시야 중심을 벗어나면 가중치 감소)
        view_angle = self.get_angle_to_firing_position(self.f16, aim)
        R_viewangle = max(0.0, 1.0 - abs(view_angle) / 90.0)  # 0~1 정규화

        R_tactical = (
            -0.001 * R_distance +
             0.002 * R_ATA +
             0.002 * R_turnrate +
             0.001 * R_AA +
             0.001 * R_viewangle
        )

        # 💠 종단 보상
        if is_done:
            if self.f16_alive:
                R_terminal = 1.0
            else:
                R_terminal = -1.0
        else:
            R_terminal = 0.0

        # 보상 조합
        if method == "proposed":
            reward = (
                weights['survival'] * R_terminal +
                weights['tactical'] * R_tactical +
                weights['entropy'] * R_entropy +
                weights['shaping'] * R_shaping
            )
        elif method == "baseline":
            reward = R_terminal
        elif method == "shaping":
            reward = R_terminal + weights['shaping'] * R_shaping
        elif method == "entropy":
            reward = R_terminal + weights['entropy'] * R_entropy
        else:
            reward = R_terminal

        # ✔️ 내부 저장 (TensorBoard 기록용)
        self.last_reward_components = {
            'reward_survival': R_terminal,
            'reward_tactical': R_tactical,
            'reward_entropy': R_entropy,
            'reward_shaping': R_shaping
        }
        return reward
    
    #########추가한거!!!!!!!!!!
    def get_closing_velocity(self, f16, aim):
        rel_vel = np.array(aim.vel0_vec) - np.array([
            f16.get_v_north(), f16.get_v_east(), f16.get_v_down()
        ])
        rel_pos = np.array(self.get_relative_unit_position_NED(f16, aim))
        rel_dist = np.linalg.norm(rel_pos)
        if rel_dist == 0:
            return 0
        return np.dot(rel_vel, rel_pos) / rel_dist

    def is_done(self):
        
        for key in self.aim_block:

            if self.aim_block[key].is_target_hit():
                self.f16_alive = False
#                self.reward_f16_dead = 1
                print('f16 Dead')
                return True
                          
            if self.aim_block[key].is_traget_lost():
                if self.aim_block[key].is_alt_low():
                    print('Missile hit ground')
                else:
                    print('f16 outrun')
                return True

        if self.f16.get_altitude() < 1e3:
#                self.reward_f16_hit_ground = 1
                print('F16 hit ground')
                return True 


        lost = [self.aim_block[key].is_traget_lost() for key in self.aim_block_names]
        if all(lost):
            print('All missiles lost')
#            self.reward_all_lost = 1
            return True

        if self.f16.get_sim_time_sec() > self.conf.general['sim_time_max']:
            print('Max time', self.f16.get_sim_time_sec())
#            self.reward_max_time = 1
            if np.isnan(self.evasion_time): # 아직 기록되지 않았을 때만(최초 성공 시점) 기록
                self.evasion_time = self.f16.get_sim_time_sec()
            return True
        else:
            return False

    def step(self, action, action_type):
        #if  PPO continues action -1 to 1
        for _ in self.r_step:
            # integrate f16 
            self.f16.step_BVR(action, action_type=action_type)

            f16_pos = (self.f16.get_lat_gc_deg(), self.f16.get_long_gc_deg(), self.f16.get_altitude())
            self.trajectory_log['f16'].append(f16_pos)

            for key, aim in self.aim_block.items():
                if not aim.is_traget_lost():
                    aim_pos = (aim.get_lat_gc_deg(), aim.get_long_gc_deg(), aim.get_altitude())
                    self.trajectory_log['aims'][key].append(aim_pos)

            for key in self.aim_block:
                # if not lost, integrate dynamics
                if not self.aim_block[key].is_traget_lost():
                    self.aim_block[key].step_evasive()

        # 최소 접근 거리(CPA) 업데이트
        for key in self.aim_block:
            # 거리를 계산하는 기존 함수를 활용합니다.
            current_dist = self.get_distance_to_firing_position(self.f16, self.aim_block[key])
            if current_dist < self.min_cpa:
                self.min_cpa = current_dist
        
        done = self.is_done()
        reward = self.get_reward(done)
        self.update_states()

        info = {}
        if done:
            info = {
                'survival': 1.0 if self.f16_alive else 0.0,
                'min_cpa': self.min_cpa,
                'evasion_time': self.evasion_time  # 성공 시 시간, 실패 시 nan
            }
        return self.f16.state_block, reward, done, info
               
    def reset_count(self):
        self.count = 0

    def reset_health(self):
        self.f16_alive = True   

    def reset_reward(self):
        self.dist_min = None
        #self.reward_f16_dead = 0
        #self.reward_aim_hit_ground = 0
        #self.reward_f16_hit_ground = 0
        #self.reward_all_lost = 0
        #self.reward_max_time = 0
        self.min_cpa = float('inf')  # 이번 에피소드의 최소 접근 거리 (무한대로 초기화)
        self.evasion_time = float('nan') # 회피 성공 시 걸린 시간 (초기값은 Not a Number)

    def reset(self, rand_state_f16 = False, rand_state_aim = False):
        
        self.reset_count()

        self.reset_health()

        self.reset_reward()

        lat0, long0, alt0, vel0, heading0 = self.get_init_state_F16(rand_state_f16)
        #print(lat0, long0, alt0, vel0, heading0)
        # deg , deg , meters, m/s, deg 
        self.f16.reset(lat0, long0, alt0, vel0, heading0)
        self.trajectory_log = {'f16': [], 'aims': {key: [] for key in self.aim_block}}

  
        for key in self.aim_block:
            # hard reset 
            if self.aim_block[key].value_error:
                for i in self.aim_block_names:            
                        fg = False
                        fg_out = None
                        from jsb_gym.TAU.config import aim_evs_BVRGym
                        self.aim_block[i] = AIM(aim_evs_BVRGym, fg, fg_out)
            
            lat, long, alt, vel, heading = self.get_init_state_AIM(key, lat0, long0, rand_state=rand_state_aim)
            self.aim_block[key].reset(lat, long, alt, vel,heading)
            self.aim_block[key].reset_target(self.f16, set_active=True)
    
        self.update_states()    
        return self.f16.state_block
    