from ml_collections import config_dict

cfg = config_dict.ConfigDict()

# more standard hyperparams:
cfg.batch_size = 1024
cfg.episodes = 50000
cfg.discount = 0.99
cfg.lr = 3e-4
cfg.expectile = 0.7
cfg.seed = 73556252

cfg.low_alpha = 3
cfg.high_alpha = 3
cfg.eval_temp = 1
cfg.tau = 0.005

# nn hyperparams
cfg.value_HD = (512, 512, 512)
cfg.low_actor_HD = (512, 512, 512)
cfg.high_actor_HD = (512, 512, 512)
cfg.const_std = True

# goal-sampling scheme hyperparams:
cfg.V_p_curgoal = 0.2  #
cfg.V_p_trajgoal = 0.5  #
cfg.V_p_randomgoal = 0.3  #
cfg.V_geom_sample = True  #

cfg.A_geom_sample = False  #
cfg.A_p_randomgoal = 0  #
cfg.subgoal_steps = 25  #
cfg.gc_negative = True  #
cfg.num_ensemble = 2
