import torch
import torch.nn as nn
import torch.nn.functional as F

class PutOffHead(nn.Module):
    """
    根据解码器状态和当前选择的工件 为每台机器预测一个put-off 延迟时段 决策。
    """
    def __init__(self, rnn_out_dim, job_embed_dim, hidden_dim, num_machines, max_delay_options):
        """
        Args:
            rnn_out_dim (int): 解码器主RNN的输出维度 (rnn_out_i)。
            job_embed_dim (int): 编码后的工件嵌入维度 (p̃_j)。
            hidden_dim (int): 此预测头内部MLP的隐藏层维度。
            num_machines (int): 机器数量。
            max_delay_options (int): 延迟决策的选项数。例如，如果可以延迟{0, 1, 2}个时段，则此值为3。
        """
        super(PutOffHead, self).__init__()
        self.num_machines = num_machines
        self.max_delay_options = max_delay_options
        
        self.mlp = nn.Sequential(
            nn.Linear(rnn_out_dim + job_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_machines * max_delay_options)
        )

    def forward(self, rnn_out, selected_job_embed):
        """
        Args:
            rnn_out (torch.Tensor): 解码器RNN的当前输出, shape: (B, rnn_out_dim)
            selected_job_embed (torch.Tensor): 被选中工件的嵌入, shape: (B, job_embed_dim)
        
        Returns:
            torch.Tensor: 每个机器上、每个延迟选项的对数概率, shape: (B, M, max_delay_options)
        """
        # 将两个输入拼接作为MLP的输入
        combined_input = torch.cat((rnn_out, selected_job_embed), dim=1)
        
        # (B, num_machines * max_delay_options)
        logits = self.mlp(combined_input)
        
        # 变形为 (B, M, max_delay_options) 以便在最后一个维度应用softmax
        logits = logits.view(-1, self.num_machines, self.max_delay_options)
        
        # 返回对数概率，用于计算actor loss
        return F.log_softmax(logits, dim=-1)