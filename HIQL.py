from loss import total_loss
from batch_gen import gen_batch
from networks.networks import LowPolicy, HighPolicy, ValueNet, GoalRep
import optax 

class HIQLAgent:
    def __init__(self):
        pass

    

    def create(self):
        self.value_net = ValueNet()
        self.target_net = ValueNet()
        self.low_lvl_policy = LowPolicy()
        self.high_lvl_policy = HighPolicy()
        self.goal_rep = GoalRep()