import torch
import torch.nn as nn

class PartI_JobEncoder(nn.Module):
    def __init__(self, scalar_input_dim=1, embedding_dim=64, hidden_dim=128, rnn_type='RNN', num_rnn_layers=1):
        super(PartI_JobEncoder, self).__init__()
        
        self.scalar_input_dim = scalar_input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type.upper()
        self.num_rnn_layers = num_rnn_layers
        
        self.embedding = nn.Linear(scalar_input_dim, embedding_dim)
        
        if self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size=embedding_dim,
                              hidden_size=hidden_dim,
                              num_layers=num_rnn_layers,
                              batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=embedding_dim,
                               hidden_size=hidden_dim,
                              num_layers=num_rnn_layers,
                              batch_first=True)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=embedding_dim,
                              hidden_size=hidden_dim,
                              num_layers=num_rnn_layers,
                              batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {self.rnn_type}. Choose from 'RNN', 'LSTM', or 'GRU'.")
        
    def forward(self, job_processing_times, h_0=None, c_0=None):
        """
            Forward pass for Part I Job Encoder.

            Args:
                job_processing_times (Tensor): Tensor of processing times for a batch of jobs.
                                            Shape: (batch_size, num_machines, scalar_input_dim)
                                            Example: (num_jobs_in_instance, m_machines, 1)
                h_0 (Tensor, optional): Initial hidden state for RNN/GRU.
                                        Shape: (num_rnn_layers * num_directions, batch_size, hidden_dim)
                                        Defaults to zeros if None.
                c_0 (Tensor, optional): Initial cell state for LSTM.
                                        Shape: (num_rnn_layers * num_directions, batch_size, hidden_dim)
                                        Defaults to zeros if None. (Only for LSTM)

            Returns:
                Tensor: Encoded job vectors p_i.
                        Shape: (batch_size, hidden_dim)
        """
        batch_size = job_processing_times.size(0)  # num_jobs_in_instance
        # num_machines = job_processing_times.size(1)
        
        embedded_seq = self.embedding(job_processing_times)
        
        if h_0 is None and self.rnn_type in ['RNN', 'GRU']:
            h_0 = torch.zeros(self.num_rnn_layers, batch_size, self.hidden_dim).to(job_processing_times.device)
        
        if self.rnn_type == 'LSTM':
            if h_0 is None:
                h_0 = torch.zeros(self.num_rnn_layers, batch_size, self.hidden_dim).to(job_processing_times.device)
            if c_0 is None:
                c_0 = torch.zeros(self.num_rnn_layers, batch_size, self.hidden_dim).to(job_processing_times.device)
            rnn_output, (h_n, c_n) = self.rnn(embedded_seq, (h_0, c_0))
        else:
            rnn_output, h_n = self.rnn(embedded_seq, h_0)
            
        p_i = h_n[-1, :, :] # Shape: (batch_size, hidden_dim)

        return p_i
    
if __name__ == '__main__':
    num_jobs_n = 10
    num_machines_m = 5
    
    embedding_dim_k_fig = 32 
    hidden_dim_p_fig = 64

    new_scalar_input_dim = 2
    
    processing_times_feature_dim = torch.rand(num_jobs_n, num_machines_m, 1)
    energy_feature_dim = torch.rand(num_jobs_n, num_machines_m, 1) * 10
    instance_data_cat = torch.cat((processing_times_feature_dim, energy_feature_dim), dim=-1)
    actual_dim_scalar_feature_dim = instance_data_cat.shape[-1]
    
    print(f"Input processing times shape: {processing_times_feature_dim.shape}")
    print(f"Input energy feature shape: {energy_feature_dim.shape}")
    print(f"Input instance data concatenated shape: {instance_data_cat.shape}")
    print(f"Actual scalar feature dimension: {actual_dim_scalar_feature_dim}")
    print("-" * 30)
    

    print(f"Input shape (all jobs in an instance): {instance_data_cat.shape}")
    print(f" (num_jobs, num_machines, scalar_feature_dim)")
    print("-" * 30)

    print("Testing with RNN:")
    part1_encoder_rnn = PartI_JobEncoder(scalar_input_dim=new_scalar_input_dim,
                                     embedding_dim=embedding_dim_k_fig,
                                     hidden_dim=hidden_dim_p_fig,
                                     rnn_type='RNN')
    p_vectors_rnn = part1_encoder_rnn(instance_data_cat)
    print(f"Output p_i vectors shape (RNN): {p_vectors_rnn.shape}")
    print(f" (num_jobs, hidden_dim)")
    print("-" * 30)

    print("Testing with LSTM:")
    part1_encoder_lstm = PartI_JobEncoder(scalar_input_dim=new_scalar_input_dim,
                                      embedding_dim=embedding_dim_k_fig,
                                      hidden_dim=hidden_dim_p_fig,
                                      rnn_type='LSTM')
    p_vectors_lstm = part1_encoder_lstm(instance_data_cat)
    print(f"Output p_i vectors shape (LSTM): {p_vectors_lstm.shape}")
    print("-" * 30)

    print("Testing with GRU (2 layers):")
    part1_encoder_gru = PartI_JobEncoder(scalar_input_dim=new_scalar_input_dim,
                                     embedding_dim=embedding_dim_k_fig,
                                     hidden_dim=hidden_dim_p_fig,
                                     rnn_type='GRU',
                                     num_rnn_layers=2)
    

    p_vectors_gru = part1_encoder_gru(instance_data_cat)

    print(f"Output p_i vectors shape (GRU, 2 layers): {p_vectors_gru.shape}")