import torch
import torch.nn as nn
from PFSPNet_Encoder import PFSPNetEncoder
from PFSPNet_Decoder import PFSPNetDecoder

class PFSPNet(nn.Module):
    def __init__(self, encoder_args, decoder_args):
        super(PFSPNet, self).__init__()
        self.encoder = PFSPNetEncoder(**encoder_args)
        self.decoder = PFSPNetDecoder(**decoder_args)
        # ! 用来保存最后一次解码的序列
        self.last_selected_indices = None
        
    def forward(self, 
                batch_instance_processing_times,  # (B, NumJobs, NumMachines, 1)
                batch_num_machines_scalar,        # (B, 1) - 每个实例的机器数
                # decoder需要的其他参数
                external_ptr_h_state=None,  
                external_ptr_c_state=None,
                max_decode_len=None,
                teacher_forcing_ratio=0.0,
                target_sequence=None
                ):
        
        batch_size = batch_instance_processing_times.size(0)
        
        # 编码器
        # 当前的PFSPNetEncoder的forward是为单个实例设计的。
        # 为了处理批次，我们需要迭代调用它，或者修改PFSPNetEncoder使其原生支持批处理。
        encoded_jobs_list = []
        for b_idx in range(batch_size):
            # PFSPNetEncoder期望的输入:
            # instance_processing_times_single_batch: (NumJobs, NumMachines, scalar_dim)
            # num_machines_scalar_single_batch: (1,1) or scalar
            single_instance_proc_times = batch_instance_processing_times[b_idx]
            single_m_scalar = batch_num_machines_scalar[b_idx].view(1,1) # 确保是(1,1)

            enc_output_single_instance = self.encoder(single_instance_proc_times, single_m_scalar)
            encoded_jobs_list.append(enc_output_single_instance)
        
        encoded_jobs_batched = torch.stack(encoded_jobs_list, dim=0) # (B, NumJobs, EncDim)
        # ! 更高效的做法是让PFSPNetEncoder原生支持批处理。

        # 解码器
        # all_job_processing_times 参数用于解码器在每步获取选中工件的加工时间
        # 它与 batch_instance_processing_times 相同
        selected_indices, log_probs, encoded_jobs_batched = self.decoder(
            encoded_jobs_batched,
            all_job_processing_times=batch_instance_processing_times, # 原始加工时间，用于查找 P_pi_t
            num_machines_scalar=batch_num_machines_scalar,            # (B,1)
            external_ptr_h_state=external_ptr_h_state,
            external_ptr_c_state=external_ptr_c_state,
            max_decode_len=max_decode_len,
            teacher_forcing_ratio=teacher_forcing_ratio,
            target_sequence=target_sequence
        )
        
        self.last_selected_indices = selected_indices.detach().clone() # 保存最后一次解码的序列
         
        return selected_indices, log_probs, encoded_jobs_batched

if __name__ == '__main__':
    batch_size_example = 2
    num_jobs_example = 5
    num_machines_example_val = 3
    
    part1_args_enc = {
        'scalar_input_dim': 1, 'embedding_dim': 32, 'hidden_dim': 64, 'rnn_type': 'RNN', 'num_rnn_layers':1
    }
    part2_args_enc = {
        'p_vector_dim': part1_args_enc['hidden_dim'], 'm_embedding_dim': 16, 'output_dim': 48
    }
    part3_args_enc = {
        'p_tilde_dim': part2_args_enc['output_dim'], 'conv_out_channels': 96, # <--- attention_job_encoding_dim
        'conv_kernel_size': 3, 'conv_padding': 'same'
    }
    encoder_params = {'part1_args': part1_args_enc, 'part2_args': part2_args_enc, 'part3_args': part3_args_enc}

    # DecoderStep1Stage的pt_encoder (RNN1)的参数
    step1_pt_enc_args_dec = {
        'scalar_input_dim': 1, 'embedding_dim': 32, 'hidden_dim': 60, # RNN1 hidden_dim
        'rnn_type': 'RNN', 'num_rnn_layers': 1
    }
    decoder_params = {
        'step1_pt_encoder_args': step1_pt_enc_args_dec,
        'step1_m_embedding_dim': 20,
        'step1_di_output_dim': 40,
        'step1_fc_hidden_dims': [30],
        'step2_rnn2_hidden_dim': 70,
        'step2_rnn_type': 'RNN',
        'step2_num_rnn_layers': 1,
        'attention_job_encoding_dim': part3_args_enc['conv_out_channels'], # 必须与编码器输出一致
        'attention_hidden_dim': 50,
        'ptr_dim': 25 # 假设我们使用可学习的ptr向量，其原始维度为25
    }

    pfsp_model = PFSPNet(encoder_args=encoder_params, decoder_args=decoder_params)
    
    # --- 准备虚拟输入数据 ---
    # 批处理的实例加工时间: (B, NumJobs, NumMachines, 1)
    dummy_batch_proc_times = torch.rand(batch_size_example, num_jobs_example, num_machines_example_val, 1)
    # 批处理的机器数: (B, 1)
    dummy_batch_m_scalar = torch.full((batch_size_example, 1), float(num_machines_example_val))

    # PTR 状态 (如果 ptr_dim=None, 则需要提供这些)
    # L1 = step1_pt_enc_args_dec['num_rnn_layers'], H1 = step1_pt_enc_args_dec['hidden_dim']
    # dummy_ptr_h = torch.randn(step1_pt_enc_args_dec['num_rnn_layers'], batch_size_example, step1_pt_enc_args_dec['hidden_dim'])
    # dummy_ptr_c = None
    # if step1_pt_enc_args_dec['rnn_type'] == 'LSTM':
    #     dummy_ptr_c = torch.randn(step1_pt_enc_args_dec['num_rnn_layers'], batch_size_example, step1_pt_enc_args_dec['hidden_dim'])

    print(f"模型实例化完成。准备进行前向传播测试...")
    # --- PFSPNet 前向传播 ---
    # 由于 decoder_params 中设置了 ptr_dim，解码器将使用内部可学习的ptr，无需传入 external_ptr_h/c_state
    selected_job_indices, log_probabilities, encoded_jobs_from_actor = pfsp_model(
        batch_instance_processing_times=dummy_batch_proc_times,
        batch_num_machines_scalar=dummy_batch_m_scalar
        # external_ptr_h_state=dummy_ptr_h, # 如果ptr_dim=None则需要
        # external_ptr_c_state=dummy_ptr_c   # 如果ptr_dim=None且RNN1是LSTM则需要
    )

    print(f"\n--- PFSPNet 输出 ---")
    print(f"选择的工件索引 (Selected Job Indices) Shape: {selected_job_indices.shape}") # (B, NumJobs)
    print(f"选择的工件索引 (Selected Job Indices) [0]: {selected_job_indices[0]}")
    print(f"对数概率 (Log Probabilities) Shape: {log_probabilities.shape}") # (B, NumJobs, NumJobs_actions)
    print(f"编码后的工件 (Encoded Jobs for Critic) Shape: {encoded_jobs_from_actor.shape}") # (B, NumJobs, EncDim)
