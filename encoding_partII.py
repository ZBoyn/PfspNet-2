import torch
import torch.nn as nn
import torch.nn.functional as F

class PartII_MachineIntegration(nn.Module):
    def __init__(self, p_vector_dim, num_machines_scalar_dim=1, m_embedding_dim=32, fc_hidden_dim=128, output_dim=64):
        """
        Part II of PFSPNet: Integrates machine count 'm' with job vectors 'p_i'.

        Args:
            p_vector_dim (int): Dimension of input job vector p_i (from Part I).
            num_machines_scalar_dim (int): Dimension of scalar machine count 'm' (typically 1).
            m_embedding_dim (int): Dimension of the embedded machine count vector m_tilde.
            fc_hidden_dim (int): Dimension of the hidden layer in the FC network (optional, can be direct to output_dim).
                                 For simplicity, we can have one FC layer mapping to output_dim.
            output_dim (int): Dimension of the output vector p_tilde_i.
        """
        super(PartII_MachineIntegration, self).__init__()
        self.p_vector_dim = p_vector_dim
        self.m_embedding_dim = m_embedding_dim

        # Step 1: Embedding for machine count 'm'
        self.machine_embedding = nn.Linear(num_machines_scalar_dim, m_embedding_dim)

        # Step 2: Fully connected layer for [m_tilde, p_i]
        self.fc_layer = nn.Linear(m_embedding_dim + p_vector_dim, output_dim)
        
        self.relu = nn.ReLU()

    def forward(self, p_vectors, num_machines_scalar):
        """
        Forward pass for Part II.

        Args:
            p_vectors (Tensor): Job vectors from Part I.
                                Shape: (num_jobs, p_vector_dim)
            num_machines_scalar (Tensor): Scalar number of machines 'm'.
                                          Shape: (1, num_machines_scalar_dim) or just a scalar
                                          that can be broadcasted or unsqueezed.
                                          It should be a tensor for nn.Linear.

        Returns:
            Tensor: Output vectors p_tilde.
                    Shape: (num_jobs, output_dim)
        """
        num_jobs = p_vectors.size(0)

        # Step 1: Embed machine count 'm'
        if num_machines_scalar.dim() == 0:
            num_machines_scalar = torch.tensor([num_machines_scalar], dtype=torch.float32).unsqueeze(0)
        elif num_machines_scalar.dim() == 1 and num_machines_scalar.size(0) == 1 :
             num_machines_scalar = num_machines_scalar.unsqueeze(0)
        elif num_machines_scalar.dim() == 1 and num_machines_scalar.size(0) > 1 :
             pass

        m_tilde = self.machine_embedding(num_machines_scalar)

        m_tilde_expanded = m_tilde.expand(num_jobs, -1)

        # concatenated_input = torch.cat((m_tilde_expanded, p_vectors), dim=-1)
        concatenated_input = torch.cat((m_tilde_expanded, p_vectors), dim=1)

        h_tilde = self.fc_layer(concatenated_input)

        p_tilde_vectors = self.relu(h_tilde)

        return p_tilde_vectors