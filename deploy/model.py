import torch
import torch.nn as nn

class DeployedActor(nn.Module):
    """
    用于部署的Actor网络。
    结构必须与 rsl_rl_ppo_cfg.py 中定义的完全一致。
    """
    def __init__(self, num_obs, num_actions):
        super().__init__()

        # 从配置文件可知，激活函数是 ELU
        activation = nn.ELU()

        # 从配置文件可知，隐藏层维度是 [512, 256, 128]
        actor_hidden_dims = [512, 256, 128]

        actor_layers = []
        in_dim = num_obs
        for hidden_dim in actor_hidden_dims:
            actor_layers.append(nn.Linear(in_dim, hidden_dim))
            actor_layers.append(activation)
            in_dim = hidden_dim
        
        # 最后一层连接到动作空间
        actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        
        self.actor_net = nn.Sequential(*actor_layers)

    def forward(self, observations):
        """前向传播，返回确定性动作。"""
        return self.actor_net(observations)
