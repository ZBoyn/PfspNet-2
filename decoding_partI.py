import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class JobProcessingTimeEncoderForDecoder(nn.Module):
    # 处理前一个作业的处理时间
    def __init__(self, scalar_input_dim=4, embedding_dim=64, hidden_dim=128, rnn_type='RNN', num_rnn_layers=1):
        super(JobProcessingTimeEncoderForDecoder, self).__init__()
        self.embedding = nn.Linear(scalar_input_dim, embedding_dim) # 嵌入层，将标量加工时间映射到向量
        
        self.rnn_type = rnn_type
        self.num_rnn_layers = num_rnn_layers
        self.hidden_dim = hidden_dim
        
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_rnn_layers, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_rnn_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=num_rnn_layers, batch_first=True)
        else:
            raise ValueError("Unsupported RNN type: {}".format(rnn_type))
    
    def forward(self, proc_times_seq, initial_hidden_state=None, initial_cell_state=None):
        # proc_times_seq: (batch_size=1, num_machines, scalar_input_dim) - 输入的加工时间序列
        # initial_hidden_state (用于 ptr): (num_rnn_layers, 1, hidden_dim) - RNN的初始隐藏状态 (来自ptr)
        # initial_cell_state (用于 ptr, 仅LSTM): (num_rnn_layers, 1, hidden_dim) - RNN的初始细胞状态 (来自ptr)
        batch_size = proc_times_seq.size(0)
        # print("proc_times_seq shape:", proc_times_seq.shape)
        embedded_seq = self.embedding(proc_times_seq) # (1, num_machines, embedding_dim)

        if self.rnn_type == 'LSTM':
            if initial_hidden_state is None: # 应该由 ptr 提供
                h0 = torch.zeros(self.num_rnn_layers, batch_size, self.hidden_dim).to(proc_times_seq.device)
            else:
                h0 = initial_hidden_state
            if initial_cell_state is None: # LSTM 也需要来自 ptr 的 c0 或零向量
                c0 = torch.zeros(self.num_rnn_layers, batch_size, self.hidden_dim).to(proc_times_seq.device)
            else:
                c0 = initial_cell_state
            _, (h_n, _) = self.rnn(embedded_seq, (h0, c0)) # LSTM 返回 output, (h_n, c_n)
        else: # RNN / GRU
            if initial_hidden_state is None: # 应该由 ptr 提供
                h0 = torch.zeros(self.num_rnn_layers, batch_size, self.hidden_dim).to(proc_times_seq.device)
            else:
                h0 = initial_hidden_state
            _, h_n = self.rnn(embedded_seq, h0) # RNN/GRU 返回 output, h_n
            
        return h_n[-1, :, :] # 返回最后一层的最终隐藏状态: (1, hidden_dim)

class DecoderStep1Stage(nn.Module):
    # 第一阶段的解码器，用于处理 P_pi-1,j 的编码器 (RNN1) 和机器数量 m 的嵌入
    def __init__(self, pt_encoder_args, m_embedding_dim, rnn1_output_dim, di_output_dim, fc_hidden_dims=None):
        super(DecoderStep1Stage, self).__init__()
        # 用于处理 P_pi-1,j 的编码器 (RNN1)
        self.pt_encoder = JobProcessingTimeEncoderForDecoder(**pt_encoder_args)
        # 机器数量 m 的嵌入层
        self.machine_embedding = nn.Linear(1, m_embedding_dim) # m 是标量

        # 全连接层部分
        fc_input_dim = rnn1_output_dim + m_embedding_dim # 输入维度是 RNN1输出 和 m嵌入 的拼接
        layers = []
        if fc_hidden_dims: # 如果定义了中间隐藏层
            for h_dim in fc_hidden_dims:
                layers.append(nn.Linear(fc_input_dim, h_dim))
                layers.append(nn.ReLU())
                fc_input_dim = h_dim
        layers.append(nn.Linear(fc_input_dim, di_output_dim)) # 最终输出 d_i
        layers.append(nn.ReLU())
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, prev_job_proc_times, ptr_h_state, ptr_c_state, num_machines_scalar):
        # prev_job_proc_times: (1, num_total_machines, 1) - 前一个工件的加工时间序列
        # ptr_h_state: (num_rnn_layers, 1, rnn1_hidden_dim) - ptr 的 h 部分，作为 RNN1 的初始隐藏状态
        # ptr_c_state: (num_rnn_layers, 1, rnn1_hidden_dim) or None - ptr 的 c 部分 (仅LSTM)
        # num_machines_scalar: tensor([[m_val]]) - 机器数量 m (标量张量)
        
        # RNN1 处理先前选择工件的加工时间
        # print("prev_job_proc_times.shape:", prev_job_proc_times.shape)
        rnn1_output = self.pt_encoder(prev_job_proc_times, 
                                      initial_hidden_state=ptr_h_state, 
                                      initial_cell_state=ptr_c_state) # (1, rnn1_output_dim)

        # 嵌入机器数量 m
        m_embedded = self.machine_embedding(num_machines_scalar) # (1, m_embedding_dim)

        # 拼接 RNN1 输出和 m 的嵌入
        concatenated = torch.cat((rnn1_output, m_embedded), dim=1) # (1, rnn1_output_dim + m_embedding_dim)

        # 通过全连接层得到 d_i
        di_vector = self.fc_layers(concatenated) # (1, di_output_dim)
        return di_vector

if __name__ == "__main__":
    # 测试 DecoderStep1Stage
    pt_encoder_args = {
        'scalar_input_dim': 1,
        'embedding_dim': 64,
        'hidden_dim': 128,
        'rnn_type': 'RNN',  # 可以是 'RNN', 'LSTM', 'GRU'
        'num_rnn_layers': 1
    }
    decoder = DecoderStep1Stage(pt_encoder_args, m_embedding_dim=32, rnn1_output_dim=128, di_output_dim=64)
    
    prev_job_proc_times = torch.randn(1, 10, 1) # 假设有10台机器
    ptr_h_state = torch.randn(1, 1, 128) # 假设 RNN 有128维隐藏状态
    ptr_c_state = torch.randn(1, 1, 128) # LSTM 的细胞状态
    num_machines_scalar = torch.tensor([[10.0]]) # 假设有10台机器

    di_vector = decoder(prev_job_proc_times, ptr_h_state, ptr_c_state, num_machines_scalar)
    print(di_vector.shape) # 应该输出 (1, 64)