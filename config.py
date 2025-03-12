from ml_collections import config_dict

cfg = config_dict.ConfigDict()

# more standard hyperparams:
cfg.discount = 0.99
cfg.lr = 10 ** -3
cfg.expectile = 0
cfg.seed=1
cfg.target_update = 50

# nn hyperparams
cfg.value_HD = (50, 50, 50)
cfg.low_actor_HD = (50, 50, 50)
cfg.high_actor_HD = (50, 50, 50)

# goal-sampling scheme hyperparams:
cfg.V_p_curgoal = 0 #
cfg.V_p_trajgoal = 0 #
cfg.V_p_randomgoal = 0 #
cfg.V_geom_sample = True #
cfg.A_geom_sample = True #
cfg.A_p_randomgoal = 0 #
cfg.subgoal_steps = 0 #
cfg.gc_negative = 0 #
cfg.num_ensemble = 2