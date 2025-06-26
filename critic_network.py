import torch
import torch.nn as nn

class CriticNetwork(nn.Module):
    def __init__(self,
                 encoder_output_dim,       # PFSPNetEncoder 输出的单个工件编码维度 (即 part3_conv_out_channels)
                 conv_channels_list,       # 3个卷积层各自的输出通道数，例如: [128, 256, 128]
                 conv_kernel_sizes_list,   # 3个卷积层各自的卷积核大小，例如: [3, 3, 3]
                 conv_strides_list=None,   # 3个卷积层各自的步长，例如: [1, 1, 1]
                 final_fc_hidden_dims=None # 可选：汇总后的全连接层隐藏维度列表, e.g., [64]
                                           # 如果为 None, 则直接从汇总特征线性映射到输出1
                ):
        super(CriticNetwork, self).__init__()

        if not isinstance(conv_channels_list, list) or len(conv_channels_list) != 3:
            print(f"Warning: Paper suggests 3 conv layers for critic. Received channels list: {conv_channels_list}")
        if not isinstance(conv_kernel_sizes_list, list) or len(conv_kernel_sizes_list) != 3:
            print(f"Warning: Paper suggests 3 conv layers for critic. Received kernel sizes list: {conv_kernel_sizes_list}")

        if conv_strides_list is None:
            conv_strides_list = [1] * len(conv_channels_list)
        elif not isinstance(conv_strides_list, list) or len(conv_strides_list) != 3:
             print(f"Warning: Paper suggests 3 conv layers for critic. Received strides list: {conv_strides_list}")


        if not (len(conv_channels_list) == len(conv_kernel_sizes_list) == len(conv_strides_list)):
            raise ValueError("conv_channels_list, conv_kernel_sizes_list, and conv_strides_list must have the same length.")

        conv_layers_seq = []
        # current_channels 是当前层期望的输入通道数
        current_channels = encoder_output_dim 

        for i in range(len(conv_channels_list)):
            out_channels = conv_channels_list[i]
            kernel_size = conv_kernel_sizes_list[i]
            stride = conv_strides_list[i]
            
            # 计算padding以尽可能保持序列长度不变 (当 stride=1 时为 'same' padding)
            # L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
            # For stride=1, dilation=1, to get L_out = L_in, padding = (kernel_size-1)//2
            padding = (kernel_size - 1) // 2
            
            conv_layers_seq.append(nn.Conv1d(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ))
            conv_layers_seq.append(nn.ReLU())
            current_channels = out_channels # 更新下一层的输入通道数
        
        self.conv_layers = nn.Sequential(*conv_layers_seq)
        
        # 记录最后一个卷积层的输出通道数，它将是后续FC层的输入维度
        self.last_conv_output_channels = current_channels

        # 定义汇总后的全连接层
        fc_layers_list = []
        input_dim_for_current_fc = self.last_conv_output_channels
        if final_fc_hidden_dims and isinstance(final_fc_hidden_dims, list):
            for hidden_dim in final_fc_hidden_dims:
                fc_layers_list.append(nn.Linear(input_dim_for_current_fc, hidden_dim))
                fc_layers_list.append(nn.ReLU())
                input_dim_for_current_fc = hidden_dim
        
        fc_layers_list.append(nn.Linear(input_dim_for_current_fc, 1)) # 最终输出一个标量基线值
        self.output_fc = nn.Sequential(*fc_layers_list)

    def forward(self, encoded_jobs):
        """
        Args:
            encoded_jobs (torch.Tensor): 来自 PFSPNetEncoder 的编码工件。
                                         Shape: (batch_size, num_jobs, encoder_output_dim)
        Returns:
            torch.Tensor: 每个实例的基线值估计。
                          Shape: (batch_size, 1)
        """
        # nn.Conv1d 期望输入形状: (batch_size, in_channels, sequence_length)
        # 当前 encoded_jobs: (batch_size, num_jobs, encoder_output_dim)
        # 需要转换为: (batch_size, encoder_output_dim, num_jobs)
        x = encoded_jobs.permute(0, 2, 1)
        
        x = self.conv_layers(x)
        # 经过卷积层后, x shape: (batch_size, self.last_conv_output_channels, sequence_length_after_convs)
        
        # "Then a layer is added to sum the output of the last convolution layer"
        # 我们将此理解为在序列长度维度上求和 (dim=2)
        x_summed = torch.sum(x, dim=2)
        # x_summed shape: (batch_size, self.last_conv_output_channels)
        
        baseline_value = self.output_fc(x_summed)
        # baseline_value shape: (batch_size, 1)
        
        return baseline_value

if __name__ == '__main__':
    # 假设的编码器输出维度 (例如 PFSPNetEncoder的 part3_conv_out_channels)
    encoder_out_dim = 96 
    
    # Critic 网络的卷积层参数
    critic_conv_channels = [64, 128, 64] # 三个卷积层的输出通道
    critic_conv_kernels = [3, 3, 3]    # 对应的卷积核大小
    critic_conv_strides = [1, 1, 1]    # 对应的步长

    # Critic 网络汇总后的全连接层参数（可选）
    critic_fc_hidden = [32]

    critic_model = CriticNetwork(
        encoder_output_dim=encoder_out_dim,
        conv_channels_list=critic_conv_channels,
        conv_kernel_sizes_list=critic_conv_kernels,
        conv_strides_list=critic_conv_strides,
        final_fc_hidden_dims=critic_fc_hidden
    )

    batch_s = 5  # 训练批次大小
    num_j = 10   # 工件数量 (序列长度)
    dummy_encoded_input = torch.randn(batch_s, num_j, encoder_out_dim)

    # 前向传播
    baseline_output = critic_model(dummy_encoded_input)

    print(f"Critic Network Input Shape: {dummy_encoded_input.shape}")
    print(f"Critic Network Output (Baseline) Shape: {baseline_output.shape}") # 预期: (batch_s, 1)
    print(f"Baseline values for batch:\n{baseline_output}")

    # 检查参数数量 (可选)
    total_params = sum(p.numel() for p in critic_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters in CriticNetwork: {total_params}")