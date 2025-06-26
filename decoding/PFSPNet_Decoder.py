import torch
import torch.nn as nn
import torch.nn.functional as F

from decoding.decoding_partI import DecoderStep1Stage, JobProcessingTimeEncoderForDecoder
from decoding.decoding_partII import DecoderStep2Stage
from decoding.decoding_attention import AttentionModule, SoftmaxModule

class PFSPNetDecoder(nn.Module):
    def __init__(self,
                 step1_pt_encoder_args, step1_m_embedding_dim, step1_di_output_dim,
                 step2_rnn2_hidden_dim, attention_job_encoding_dim, attention_hidden_dim,
                 step1_fc_hidden_dims=None,
                 step2_rnn_type='RNN', step2_num_rnn_layers=1,
                 # ptr_mode: "learnable_static", "learnable_per_instance", "fixed_input"
                 ptr_dim=None, # 如果 ptr 是可学习的，这是其原始维度，否则是输入ptr的预期维度
                 ):
        super(PFSPNetDecoder, self).__init__()

        self.step1_rnn1_is_lstm = step1_pt_encoder_args['rnn_type'] == 'LSTM'
        self.step1_rnn1_num_layers = step1_pt_encoder_args.get('num_rnn_layers', 1)
        self.step1_rnn1_hidden_dim = step1_pt_encoder_args['hidden_dim']
        
        self.step2_rnn2_is_lstm = step2_rnn_type == 'LSTM'
        self.step2_num_rnn_layers = step2_num_rnn_layers
        self.step2_rnn2_hidden_dim = step2_rnn2_hidden_dim

        self.step1_stage = DecoderStep1Stage(
            pt_encoder_args=step1_pt_encoder_args,
            m_embedding_dim=step1_m_embedding_dim,
            rnn1_output_dim=self.step1_rnn1_hidden_dim, # 确保与pt_encoder的hidden_dim一致
            di_output_dim=step1_di_output_dim,
            fc_hidden_dims=step1_fc_hidden_dims
        )
        self.step2_stage = DecoderStep2Stage(
            di_input_dim=step1_di_output_dim,
            rnn2_hidden_dim=self.step2_rnn2_hidden_dim,
            rnn_type=step2_rnn_type,
            num_rnn_layers=self.step2_num_rnn_layers
        )
        self.attention_module = AttentionModule(
            job_encoding_dim=attention_job_encoding_dim, # 来自编码器的输出维度
            rnn_output_dim=self.step2_rnn2_hidden_dim,
            attention_hidden_dim=attention_hidden_dim
        )
        self.softmax_module = SoftmaxModule(dim=-1)

        # 可学习的 d_0^* (RNN2 的初始状态)
        self.initial_d0_star_h = nn.Parameter(torch.randn(self.step2_num_rnn_layers, 1, self.step2_rnn2_hidden_dim) * 0.1)
        if self.step2_rnn2_is_lstm:
            self.initial_d0_star_c = nn.Parameter(torch.randn(self.step2_num_rnn_layers, 1, self.step2_rnn2_hidden_dim) * 0.1)
        else:
            self.register_parameter('initial_d0_star_c', None)

        # ptr 的处理: 假设 ptr 是一个可学习的静态向量，用于所有实例和步骤
        # 论文描述为 "k-dimensional parameter vector"
        if ptr_dim is not None: # 如果ptr是模块内部可学习的
            self.ptr_raw_learnable = nn.Parameter(torch.randn(1, ptr_dim) * 0.1)
            # 将原始ptr投影到RNN1所需的 h 和 c 状态
            self.ptr_h_projection = nn.Linear(ptr_dim, self.step1_rnn1_num_layers * self.step1_rnn1_hidden_dim)
            if self.step1_rnn1_is_lstm:
                self.ptr_c_projection = nn.Linear(ptr_dim, self.step1_rnn1_num_layers * self.step1_rnn1_hidden_dim)
            else:
                self.ptr_c_projection = None
        else: # ptr 将从外部传入
            self.ptr_raw_learnable = None


    def forward(self,
                encoded_jobs,               # (B, NumJobs, EncDim) - 来自 PFSPNetEncoder
                all_job_processing_times,   # (B, NumJobs, NumMachines, 1) - 原始加工时间，用于查找 P_pi_t
                num_machines_scalar,        # (B, 1) - 机器数量m (每个实例可以不同)
                # 如果ptr不是内部可学习的，则需要从外部传入ptr的初始状态
                # 否则，如果ptr_raw_learnable存在，则忽略这些参数
                external_ptr_h_state=None,    # (L1, B, H1)
                external_ptr_c_state=None,    # (L1, B, H1)
                max_decode_len=None,
                teacher_forcing_ratio=0.0,
                target_sequence=None        # (B, MaxLen) - 整数索引
               ):
        batch_size, num_total_jobs, _ = encoded_jobs.shape
        device = encoded_jobs.device
        num_machines = all_job_processing_times.size(2)

        if max_decode_len is None:
            max_decode_len = num_total_jobs

        outputs_indices = []
        outputs_log_probs = [] # 存储完整概率分布的log

        # --- PTR 状态准备 ---
        ptr_h_for_step1, ptr_c_for_step1 = None, None
        if self.ptr_raw_learnable is not None:
            ptr_batch_expanded = self.ptr_raw_learnable.expand(batch_size, -1) # (B, ptr_dim)
            ptr_h_flat = self.ptr_h_projection(ptr_batch_expanded) # (B, L1*H1)
            ptr_h_for_step1 = ptr_h_flat.view(batch_size, self.step1_rnn1_num_layers, self.step1_rnn1_hidden_dim).permute(1,0,2).contiguous() # (L1,B,H1)
            if self.step1_rnn1_is_lstm:
                ptr_c_flat = self.ptr_c_projection(ptr_batch_expanded) # (B, L1*H1)
                ptr_c_for_step1 = ptr_c_flat.view(batch_size, self.step1_rnn1_num_layers, self.step1_rnn1_hidden_dim).permute(1,0,2).contiguous()
        elif external_ptr_h_state is not None:
            ptr_h_for_step1 = external_ptr_h_state
            ptr_c_for_step1 = external_ptr_c_state
        else:
            # 如果没有可学习的ptr，也没有外部传入，则默认为零（或报错）
            ptr_h_for_step1 = torch.zeros(self.step1_rnn1_num_layers, batch_size, self.step1_rnn1_hidden_dim, device=device)
            if self.step1_rnn1_is_lstm:
                ptr_c_for_step1 = torch.zeros(self.step1_rnn1_num_layers, batch_size, self.step1_rnn1_hidden_dim, device=device)
        
        # --- 解码循环初始化 ---
        # 当前认为的"上一个工件"的加工时间 (P_pi_0,j = 0)
        # ! 修改维度... ...
        current_P_prev_job = torch.zeros(batch_size, num_machines, 2, device=device)
        # 当前 RNN2 的状态 (d_0^*)
        current_rnn2_h = self.initial_d0_star_h.expand(-1, batch_size, -1).contiguous()
        current_rnn2_c = None
        if self.step2_rnn2_is_lstm:
            current_rnn2_c = self.initial_d0_star_c.expand(-1, batch_size, -1).contiguous()

        job_availability_mask = torch.ones(batch_size, num_total_jobs, device=device, dtype=torch.bool)
        batch_indices = torch.arange(batch_size, device=device) # 用于高级索引

        for t in range(max_decode_len):
            # 1. DecoderStep1Stage
            # num_machines_scalar 应该是 (B,1)
            
            di_vec = self.step1_stage(current_P_prev_job, ptr_h_for_step1, ptr_c_for_step1, num_machines_scalar.view(batch_size, 1))

            # 2. DecoderStep2Stage
            rnn_out_star, rnn2_state_next = self.step2_stage(di_vec, current_rnn2_h, current_rnn2_c)
            
            # 3. AttentionModule
            attn_scores = self.attention_module(encoded_jobs, rnn_out_star)
            
            # 4. SoftmaxModule
            probs = self.softmax_module(attn_scores, job_availability_mask) # (B, NumJobs)
            log_probs = torch.log(probs + 1e-9) # 避免 log(0)
            outputs_log_probs.append(log_probs)

            # 5. 选择工件
            chosen_job_idx = None
            if self.training and torch.rand(1).item() < teacher_forcing_ratio and target_sequence is not None:
                chosen_job_idx = target_sequence[:, t] # (B,)
            else:
                chosen_job_idx = torch.multinomial(probs, 1).squeeze(1) # 采样
                # chosen_job_idx = torch.argmax(probs, dim=1) # Greedy
            outputs_indices.append(chosen_job_idx.unsqueeze(1)) # (B,1)

            # 6. 更新状态为下一次迭代准备
            # 更新 P_prev_job (使用原始加工时间数据)
            current_P_prev_job = all_job_processing_times[batch_indices, chosen_job_idx] # (B, NumMachines, 1)
            
            # 更新 RNN2 state
            if self.step2_rnn2_is_lstm:
                current_rnn2_h, current_rnn2_c = rnn2_state_next
            else:
                current_rnn2_h = rnn2_state_next
            
            job_availability_mask[batch_indices, chosen_job_idx] = False
            
            if not job_availability_mask.any(dim=1).all() and t < max_decode_len -1 :
                 pass


        final_indices = torch.cat(outputs_indices, dim=1)
        final_log_probs = torch.stack(outputs_log_probs, dim=1)

        return final_indices, final_log_probs, encoded_jobs
    
if __name__ == '__main__':
    # 测试 PFSPNetDecoder 的基本功能
    batch_size = 2
    num_jobs = 5
    num_machines = 3
    enc_dim = 4
    di_output_dim = 6
    rnn2_hidden_dim = 8

    # 模拟编码器输出
    encoded_jobs = torch.randn(batch_size, num_jobs, enc_dim)
    all_job_processing_times = torch.randn(batch_size, num_jobs, num_machines, 2)
    num_machines_scalar = torch.tensor([[float(num_machines)]] * batch_size)

    decoder = PFSPNetDecoder(
        step1_pt_encoder_args={'rnn_type': 'GRU', 'hidden_dim': 10, 'scalar_input_dim': 2, 'embedding_dim': 64},
        step1_m_embedding_dim=5,
        step1_di_output_dim=di_output_dim,
        step2_rnn2_hidden_dim=rnn2_hidden_dim,
        attention_job_encoding_dim=enc_dim,
        attention_hidden_dim=12,
        step1_fc_hidden_dims=[20, 15],
        step2_rnn_type='GRU',
        step2_num_rnn_layers=2,
        ptr_dim=3
    )

    selected_indices, log_probs, encoded_jobs_out = decoder(
        encoded_jobs,
        all_job_processing_times,
        num_machines_scalar,
        max_decode_len=num_jobs
    )

    print("Selected Indices:", selected_indices)
    print("Log Probabilities:", log_probs)
    print("Encoded Jobs Output Shape:", encoded_jobs_out.shape)