import numpy as np

version = 1

aim1 = {'bearing': 0,
        'distance': 90e3,
        'vel': 300,
        'alt': 12e3}

aim = {'aim1': aim1}

# random location expects a list 
aim1_rand = {'bearing':  [0, 360],
             'distance': [40e3, 100e3],
             'vel':      [290, 340],
             'alt':      [10e3, 12e3]}

aim_rand = {'aim1': aim1_rand}



general = {
        'env_name': 'evasive',
        'f16_name': 'f16',
        'sim_time_max': 60*4,           
        'r_step' : 5,
        'fg_r_step' : 1,
        'missile_idle': False,
        'scale': True,
        'rec_f16': True,
        'rec_aim': True}

states= {
        'obs_space': 10,
        'act_space': 2,
        'update_states_type': 1
}

sf = {  'd_min': 60e3,
        'd_max': 120e3,
        't': general['sim_time_max'],
        'mach_max': 2,
        'alt_min': 3e3,
        'alt_max': 14e3,
        'd_max_reward': 20e3,
        'aim_vel0_min': 280, 
        'aim_vel0_max': 320, 
        'aim_alt0_min': 9e3, 
        'aim_alt0_max': 11e3}


f16 = { 'lat':      59.0,
        'long':     18.0,
        'vel':      300,
        'alt':      10e3,
        'heading' : 0}


f16_rand = {'lat':      [59.0, 59.0],
            'long':     [18.0, 18.0],
            'vel':      [250,330],
            'alt':      [6e3, 12e3],
            'heading' : [0, 2*np.pi]}

logs= {'log_path': '/home/edvards/workspace/jsbsim/jsb_gym/logs/BVRGym',
       'save_to':'/home/edvards/workspace/jsbsim/jsb_gym/plots/BVRGym'}