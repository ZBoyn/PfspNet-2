import torch
import torch.nn as nn

class PartIII_Convolutional(nn.Module):
    def __init__(self, p_tilde_dim, conv_out_channels, conv_kernel_size=3, conv_padding='same'):
        """
            Part III of PFSPNet: 1D Convolution over job vectors p_tilde_i.

            Args:
                p_tilde_dim (int): Dimension of input vectors p_tilde_i (from Part II).
                                This is 'in_channels' for Conv1d.
                conv_out_channels (int): Dimension of output vectors p_bar_i.
                                        This is 'out_channels' for Conv1d.
                conv_kernel_size (int): Kernel size for Conv1d.
                conv_padding (str or int): Padding for Conv1d. 'same' tries to keep seq length.
        """
        super(PartIII_Convolutional, self).__init__()

        # 1D Convolutional Layer
        self.conv1d = nn.Conv1d(in_channels=p_tilde_dim,
                                out_channels=conv_out_channels,
                                kernel_size=conv_kernel_size,
                                padding=conv_padding) 
        
        self.relu = nn.ReLU()

    def forward(self, p_tilde_vectors_sequence):
        """
            Forward pass for Part III.

            Args:
                p_tilde_vectors_sequence (Tensor): Sequence of p_tilde vectors from Part II.
                                                Shape: (num_jobs, p_tilde_dim)

            Returns:
                Tensor: Output vectors p_bar.
                        Shape: (num_jobs, conv_out_channels)
        """
        input_for_conv = p_tilde_vectors_sequence.unsqueeze(0).transpose(1, 2)
        conv_output = self.conv1d(input_for_conv)

        activated_output = self.relu(conv_output)
        p_bar_vectors = activated_output.squeeze(0).transpose(0, 1)

        return p_bar_vectors