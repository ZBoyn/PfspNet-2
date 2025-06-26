import torch
import torch.nn as nn
from encoding_partI import PartI_JobEncoder
from encoding_partII import PartII_MachineIntegration
from encoding_partIII import PartIII_Convolutional

class PFSPNetEncoder(nn.Module):
    def __init__(self, part1_args, part2_args, part3_args):
        """
        Complete PFSPNet Encoder (Part I + Part II + Part III).

        Args:
            part1_args (dict): Arguments for PartI_JobEncoder.
                               e.g., {'scalar_input_dim':1, 'embedding_dim':64, 'hidden_dim':128, 'rnn_type':'RNN'}
            part2_args (dict): Arguments for PartII_MachineIntegration.
                               e.g., {'p_vector_dim':128, 'm_embedding_dim':32, 'output_dim':64}
                               (p_vector_dim must match part1's hidden_dim)
            part3_args (dict): Arguments for PartIII_Convolutional.
                               e.g., {'p_tilde_dim':64, 'conv_out_channels':128, 'conv_kernel_size':3}
                               (p_tilde_dim must match part2's output_dim)
        """
        super(PFSPNetEncoder, self).__init__()

        self.part1_encoder = PartI_JobEncoder(**part1_args)
        
        # Ensure dimensions match between parts
        if part1_args['hidden_dim'] != part2_args['p_vector_dim']:
            raise ValueError("Output dimension of Part I (hidden_dim) must match "
                             "input dimension of Part II (p_vector_dim).")
        self.part2_integrator = PartII_MachineIntegration(**part2_args)

        if part2_args['output_dim'] != part3_args['p_tilde_dim']:
            raise ValueError("Output dimension of Part II (output_dim) must match "
                             "input dimension of Part III (p_tilde_dim).")
        self.part3_convoluter = PartIII_Convolutional(**part3_args)

    def forward(self, instance_processing_times, num_machines_scalar):
        """
        Forward pass for the complete PFSPNet Encoder.

        Args:
            instance_processing_times (Tensor): Processing times for all jobs in an instance.
                                                Shape: (num_jobs, num_machines_in_job_seq, scalar_input_dim)
            num_machines_scalar (Tensor or float): Scalar number of machines 'm' for this instance.
                                                   Shape: (1,1) or scalar.

        Returns:
            Tensor: Final encoded job vectors p_bar.
                    Shape: (num_jobs, part3_conv_out_channels)
        """
        # Part I: Encode each job's processing time sequence
        # Input: (num_jobs, num_machines_in_job_seq, 1)
        # Output p_vectors: (num_jobs, part1_hidden_dim)
        p_vectors = self.part1_encoder(instance_processing_times)

        # Part II: Integrate machine count 'm'
        # Input p_vectors: (num_jobs, part1_hidden_dim)
        # Input num_machines_scalar: e.g., tensor([[5.]])
        # Output p_tilde_vectors: (num_jobs, part2_output_dim)
        p_tilde_vectors = self.part2_integrator(p_vectors, num_machines_scalar)

        # Part III: 1D Convolution over the sequence of p_tilde_vectors
        # Input p_tilde_vectors: (num_jobs, part2_output_dim)
        # Output p_bar_vectors: (num_jobs, part3_conv_out_channels)
        p_bar_vectors = self.part3_convoluter(p_tilde_vectors)

        return p_bar_vectors

if __name__ == '__main__':
    num_jobs_n = 10
    num_machines_m_val = 5
    
    
    part1_embedding_dim = 32
    part1_hidden_dim = 64
    part1_rnn_type = 'RNN'

    part2_p_vector_dim = part1_hidden_dim # Must match part1_hidden_dim
    part2_m_embedding_dim = 16
    part2_fc_output_dim = 48 # p_tilde_i dim

    part3_p_tilde_dim = part2_fc_output_dim # Must match part2_fc_output_dim
    part3_conv_out_channels = 96 # p_bar_i dim
    part3_conv_kernel_size = 3   # Example kernel size
    part3_conv_padding = 'same'  # To keep num_jobs dimension same if kernel is odd


    # Create sample input for one instance  
    # (num_jobs, num_machines_in_job_seq, scalar_feature_dim)
    proc_times = torch.rand(num_jobs_n, num_machines_m_val, 1)
    
    # ! add new scalar input dimension
    # Energy feature dimension
    energy_scalar_feature_dim = torch.rand(num_jobs_n, num_machines_m_val, 1) * 10
    # 对 proc_times 中的每个值当启用高功耗时为 0.7 倍
    proc_times_high_power = proc_times * 0.7
    energy_high_power = energy_scalar_feature_dim * 1.5    
    
    # 拼接上面的所有维度
    conbined_features = torch.cat((proc_times, energy_scalar_feature_dim, proc_times_high_power, energy_high_power), dim=-1)
    part1_scalar_input_dim = conbined_features.shape[-1]  # This will be 4 if we have 4 features
    
    
            
    # Prepare arguments for each part
    part1_args = {
        'scalar_input_dim': part1_scalar_input_dim,
        'embedding_dim': part1_embedding_dim,
        'hidden_dim': part1_hidden_dim,
        'rnn_type': part1_rnn_type
    }
    part2_args = {
        'p_vector_dim': part2_p_vector_dim,
        'm_embedding_dim': part2_m_embedding_dim,
        'output_dim': part2_fc_output_dim
    }
    part3_args = {
        'p_tilde_dim': part3_p_tilde_dim,
        'conv_out_channels': part3_conv_out_channels,
        'conv_kernel_size': part3_conv_kernel_size,
        'conv_padding': part3_conv_padding
    }

    # Create the complete encoder
    pfsp_encoder = PFSPNetEncoder(part1_args, part2_args, part3_args)

    
    sample_m_scalar = torch.tensor([[float(num_machines_m_val)]]) # Shape (1,1) for nn.Linear

    print(f"Input instance processing times shape: {proc_times.shape}")
    print(f"Input energy feature shape: {energy_scalar_feature_dim.shape}")
    print(f"High power processing times shape: {proc_times_high_power.shape}")
    print(f"High power energy feature shape: {energy_high_power.shape}")
    print("-" * 40)
    print(f"Combined features shape: {conbined_features.shape}")
    print("-" * 40)
    
    print(f"Input machine count m: {sample_m_scalar}")
    print("-" * 40)

    # Perform forward pass
    # final_encoded_vectors = pfsp_encoder(proc_times, sample_m_scalar)
    final_encoded_vectors = pfsp_encoder(conbined_features, sample_m_scalar)
    print("-" * 40)

    print(f"Output p_bar vectors shape: {final_encoded_vectors.shape}")
    print(f"Expected: (num_jobs, part3_conv_out_channels) = ({num_jobs_n}, {part3_conv_out_channels})")
    print("-" * 40)

    part1_module = PartI_JobEncoder(**part1_args)
    p_vecs = part1_module(conbined_features)
    print(f"Part I output shape: {p_vecs.shape}")

    part2_module = PartII_MachineIntegration(**part2_args)
    p_tilde_vecs = part2_module(p_vecs, sample_m_scalar)
    print(f"Part II output shape: {p_tilde_vecs.shape}")
    
    part3_module = PartIII_Convolutional(**part3_args)
    p_bar_vecs = part3_module(p_tilde_vecs)
    print(f"Part III output shape: {p_bar_vecs.shape}")
