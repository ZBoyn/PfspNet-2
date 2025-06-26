import torch
import torch.nn as nn
from decoding_partI import JobProcessingTimeEncoderForDecoder, DecoderStep1Stage

class DecoderStep2Stage(nn.Module):
    """
        Implements Step 2 of the decoding network:
        Inputs d_i and d_{i-1}^* to the second decoding RNN (RNN2)
        Outputs d_i^* (hidden state of RNN2) and rnn_out_i^* (output of RNN2).
    """
    def __init__(self, di_input_dim, rnn2_hidden_dim, rnn_type='RNN', num_rnn_layers=1):
        """
        Args:
            di_input_dim (int): Dimension of d_i vector from Step 1.
            rnn2_hidden_dim (int): Hidden dimension of RNN2. This is also the dimension of d_i^*
                                   and rnn_out_i^*. The d_{i-1}^* used for concatenation
                                   will also have this dimension.
            rnn_type (str): Type of RNN ('RNN', 'LSTM', 'GRU').
            num_rnn_layers (int): Number of layers for RNN2.
        """
        super(DecoderStep2Stage, self).__init__()
        self.di_input_dim = di_input_dim
        self.rnn2_hidden_dim = rnn2_hidden_dim
        self.rnn_type = rnn_type
        self.num_rnn_layers = num_rnn_layers

        rnn2_actual_input_dim = di_input_dim + rnn2_hidden_dim

        if rnn_type == 'RNN':
            self.rnn2 = nn.RNN(rnn2_actual_input_dim, rnn2_hidden_dim, num_layers=num_rnn_layers, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn2 = nn.LSTM(rnn2_actual_input_dim, rnn2_hidden_dim, num_layers=num_rnn_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn2 = nn.GRU(rnn2_actual_input_dim, rnn2_hidden_dim, num_layers=num_rnn_layers, batch_first=True)
        else:
            raise ValueError("Unsupported RNN type: {}".format(rnn_type))

    def forward(self, di_vector, prev_rnn2_h_state, prev_rnn2_c_state=None):
        """
        Args:
            di_vector (torch.Tensor): The d_i vector from DecoderStep1Stage.
                                      Shape: (batch_size, di_input_dim)
            prev_rnn2_h_state (torch.Tensor): The hidden state (h-part of d_{i-1}^*) from the
                                              previous step of this RNN2.
                                              Shape: (num_rnn_layers, batch_size, rnn2_hidden_dim)
            prev_rnn2_c_state (torch.Tensor, optional): The cell state (c-part of d_{i-1}^*) for LSTM,
                                                        from the previous step. Defaults to None.
                                                        Shape: (num_rnn_layers, batch_size, rnn2_hidden_dim)
        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor] or torch.Tensor]:
            - rnn_out_i_star (torch.Tensor): RNN output for the current step.
                                             Shape: (batch_size, rnn2_hidden_dim)
            - current_rnn2_state (Tuple or Tensor): The new hidden state d_i^*.
                - For LSTM: (current_rnn2_h_state, current_rnn2_c_state)
                - For RNN/GRU: current_rnn2_h_state
              Shapes are (num_rnn_layers, batch_size, rnn2_hidden_dim)
        """
        batch_size = di_vector.size(0)

        # d_{i-1}^* for concatenation is the hidden state from the last layer of RNN2 from the previous step
        # prev_rnn2_h_state has shape (num_rnn_layers, batch_size, rnn2_hidden_dim)
        d_star_for_concat = prev_rnn2_h_state[-1] # Shape: (batch_size, rnn2_hidden_dim)

        # Concatenate d_i and d_{i-1}^* (from prev_rnn2_h_state's last layer)
        concat_input = torch.cat((di_vector, d_star_for_concat), dim=1)
        # Shape: (batch_size, di_input_dim + rnn2_hidden_dim)

        # RNNs expect input of shape (batch_size, seq_len, input_features)
        # Here, we are processing one step at a time, so seq_len = 1.
        rnn_input_seq = concat_input.unsqueeze(1)
        # Shape: (batch_size, 1, di_input_dim + rnn2_hidden_dim)

        if self.rnn_type == 'LSTM':
            if prev_rnn2_c_state is None: # Should be provided if LSTM
                # This might happen at the very first step (i=0 for d_0^*), where it's initialized to zeros.
                prev_rnn2_c_state = torch.zeros_like(prev_rnn2_h_state).to(prev_rnn2_h_state.device)
            rnn_output_seq, (current_h_state, current_c_state) = self.rnn2(rnn_input_seq, (prev_rnn2_h_state, prev_rnn2_c_state))
            current_rnn2_state = (current_h_state, current_c_state)
        else: # RNN or GRU
            rnn_output_seq, current_h_state = self.rnn2(rnn_input_seq, prev_rnn2_h_state)
            current_rnn2_state = current_h_state

        # rnn_output_seq has shape (batch_size, 1, rnn2_hidden_dim)
        # We need rnn_out_i^* which is the output for the current single time step
        rnn_out_i_star = rnn_output_seq.squeeze(1) # Shape: (batch_size, rnn2_hidden_dim)

        return rnn_out_i_star, current_rnn2_state

if __name__ == "__main__":
    batch_s = 1
    di_dim_val = 64
    rnn2_hidden_dim_val = 128
    num_rnn2_layers = 1 
    rnn2_type = 'RNN'

    pt_encoder_args_val = {'scalar_input_dim': 1, 'embedding_dim': 32, 'hidden_dim': 64, 'rnn_type': 'RNN', 'num_rnn_layers': 1}
    step1_decoder = DecoderStep1Stage(pt_encoder_args_val, m_embedding_dim=16, rnn1_output_dim=64, di_output_dim=di_dim_val)

    num_machines = 5
    prev_job_proc_times_d1 = torch.zeros(batch_s, num_machines, 1)
    ptr_h_state_d1 = torch.randn(pt_encoder_args_val['num_rnn_layers'], batch_s, pt_encoder_args_val['hidden_dim'])
    ptr_c_state_d1 = None
    num_machines_scalar_val = torch.tensor([[float(num_machines)]])

    di_vector_output = step1_decoder(prev_job_proc_times_d1, ptr_h_state_d1, ptr_c_state_d1, num_machines_scalar_val)
    print(f"Shape of d_i from Step 1: {di_vector_output.shape}")

    step2_decoder = DecoderStep2Stage(di_input_dim=di_dim_val,
                                      rnn2_hidden_dim=rnn2_hidden_dim_val,
                                      rnn_type=rnn2_type,
                                      num_rnn_layers=num_rnn2_layers)

    # For the very first call to Step 2 (calculating d_1^* and rnn_out_1^*):
    # d_0^* needs to be initialized, typically to zeros.
    prev_rnn2_h_init = torch.zeros(num_rnn2_layers, batch_s, rnn2_hidden_dim_val)
    prev_rnn2_c_init = None
    if rnn2_type == 'LSTM':
        prev_rnn2_c_init = torch.zeros(num_rnn2_layers, batch_s, rnn2_hidden_dim_val)

    rnn_out_star, current_rnn2_state_val = step2_decoder(di_vector_output, prev_rnn2_h_init, prev_rnn2_c_init)

    print(f"Shape of rnn_out_i*: {rnn_out_star.shape}") # Expected: (batch_s, rnn2_hidden_dim_val)
    if rnn2_type == 'LSTM':
        print(f"Shape of current RNN2 h-state (d_i^* h-part): {current_rnn2_state_val[0].shape}") # (num_rnn2_layers, batch_s, rnn2_hidden_dim_val)
        print(f"Shape of current RNN2 c-state (d_i^* c-part): {current_rnn2_state_val[1].shape}") # (num_rnn2_layers, batch_s, rnn2_hidden_dim_val)
    else:
        print(f"Shape of current RNN2 state (d_i^*): {current_rnn2_state_val.shape}") # (num_rnn2_layers, batch_s, rnn2_hidden_dim_val)
