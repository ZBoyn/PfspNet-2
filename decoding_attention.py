import torch
import torch.nn as nn
import torch.nn.functional as F # Not strictly needed for these modules but good to have

class AttentionModule(nn.Module):
    def __init__(self, job_encoding_dim, rnn_output_dim, attention_hidden_dim):
        """
        Args:
            job_encoding_dim (int): 单个工件编码 P_tilde_j 的维度。
            rnn_output_dim (int): 来自DecoderStep2Stage的 rnn_out_i_star 的维度。
            attention_hidden_dim (int): 注意力机制内部MLP的隐藏维度。
        """
        super(AttentionModule, self).__init__()
        self.job_encoding_dim = job_encoding_dim
        self.rnn_output_dim = rnn_output_dim
        self.attention_hidden_dim = attention_hidden_dim

        # 拼接后的维度 [P_tilde_j; rnn_out_i*]
        concat_dim = job_encoding_dim + rnn_output_dim

        # W1_layer 对应公式中的 W_1
        # 它将拼接后的特征投影到 attention_hidden_dim
        self.W1_layer = nn.Linear(concat_dim, attention_hidden_dim, bias=True)

        # v_layer 对应公式中的 v^T
        # 它将 attention_hidden_dim 投影到一个标量得分
        self.v_layer = nn.Linear(attention_hidden_dim, 1, bias=False) # v 通常不带偏置

    def forward(self, P_tilde_all_jobs, rnn_out_i_star):
        """
        Args:
            P_tilde_all_jobs (torch.Tensor): 所有工件的编码。
                                              Shape: (batch_size, num_total_jobs, job_encoding_dim)
            rnn_out_i_star (torch.Tensor): 来自 DecoderStep2Stage 的输出。
                                          Shape: (batch_size, rnn_output_dim)

        Returns:
            torch.Tensor: 每个工件的注意力得分 (logits) u_i。
                          Shape: (batch_size, num_total_jobs)
        """
        batch_size, num_total_jobs, _ = P_tilde_all_jobs.shape

        #扩展 rnn_out_i_star 以便与每个工件的编码进行拼接
        # rnn_out_i_star shape: (batch_size, rnn_output_dim)
        # expanded_rnn_out shape: (batch_size, num_total_jobs, rnn_output_dim)
        expanded_rnn_out = rnn_out_i_star.unsqueeze(1).expand(-1, num_total_jobs, -1)

        # 为所有工件拼接 P_tilde_j 和 rnn_out_i_star
        # concat_features shape: (batch_size, num_total_jobs, job_encoding_dim + rnn_output_dim)
        concat_features = torch.cat((P_tilde_all_jobs, expanded_rnn_out), dim=2)

        # 应用 W1 和 tanh
        # W1_projected shape: (batch_size, num_total_jobs, attention_hidden_dim)
        W1_projected = self.W1_layer(concat_features)
        activated_W1 = torch.tanh(W1_projected)

        # 应用 v
        # scores_ui shape: (batch_size, num_total_jobs, 1)
        scores_ui = self.v_layer(activated_W1)

        # 压缩最后一个维度得到 (batch_size, num_total_jobs)
        return scores_ui.squeeze(-1)

class SoftmaxModule(nn.Module):
    """
    实现了Softmax步骤 (Step 4)。
    可以处理mask，以便只在有效/可选的工件上计算概率。
    """
    def __init__(self, dim=-1):
        super(SoftmaxModule, self).__init__()
        self.softmax = nn.Softmax(dim=dim)
        self.dim = dim

    def forward(self, scores_ui, mask=None):
        """
        Args:
            scores_ui (torch.Tensor): 来自 AttentionModule 的注意力得分 (logits)。
                                     Shape: (batch_size, num_items)
            mask (torch.Tensor, optional): 布尔或0/1张量，指示可用的工件/项。
                                           对于不可用的项，mask中对应位置为0或False。
                                           Shape: (batch_size, num_items)

        Returns:
            torch.Tensor: 每个候选工件的概率。
                          Shape: (batch_size, num_items)
        """
        if mask is not None:
            if not isinstance(mask, torch.BoolTensor):
                mask = mask.bool()

            scores_ui_masked = scores_ui.clone()
            scores_ui_masked[~mask] = -float('inf') # ~mask 表示 mask中为False/0的位置
            return self.softmax(scores_ui_masked)
        else:
            return self.softmax(scores_ui)

if __name__ == '__main__':
    # 定义测试参数
    batch_size = 2
    num_total_jobs = 5  # 假设总共有5个工件
    job_encoding_dim = 32  # P_tilde_j 的维度
    rnn_output_dim = 64   # rnn_out_i_star 的维度 (来自DecoderStep2Stage)
    attention_hidden_dim = 48 # 注意力机制的隐藏层维度

    # 实例化模块
    attention_decoder = AttentionModule(job_encoding_dim, rnn_output_dim, attention_hidden_dim)
    softmax_decoder = SoftmaxModule(dim=-1)

    # 创建虚拟输入数据
    # P_tilde_all_jobs: 所有工件的编码 (batch_size, num_total_jobs, job_encoding_dim)
    dummy_P_tilde_all_jobs = torch.randn(batch_size, num_total_jobs, job_encoding_dim)

    # rnn_out_i_star: DecoderStep2Stage 的输出 (batch_size, rnn_output_dim)
    dummy_rnn_out_i_star = torch.randn(batch_size, rnn_output_dim)

    # --- 测试 AttentionModule ---
    attention_scores = attention_decoder(dummy_P_tilde_all_jobs, dummy_rnn_out_i_star)
    print(f"AttentionModule - Shape of results (scores_ui): {attention_scores.shape}") # 预期: (batch_size, num_total_jobs)

    # --- 测试 SoftmaxModule ---
    # 1. 无mask的情况
    probabilities_no_mask = softmax_decoder(attention_scores)
    print(f"SoftmaxModule (without mask) - Shape of results: {probabilities_no_mask.shape}") # 预期: (batch_size, num_total_jobs)
    print(f"SoftmaxModule (without mask) - Sum of probabilities (batch 0): {probabilities_no_mask[0].sum()}") # 预期: 接近 1.0

    # 2. 有mask的情况
    # 假设在batch 0中，工件0, 2, 4可用；在batch 1中，工件1, 3可用
    # Mask: 1表示可用, 0表示不可用
    dummy_mask = torch.tensor([
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0]
    ], dtype=torch.bool) # 使用布尔类型

    probabilities_with_mask = softmax_decoder(attention_scores, mask=dummy_mask)
    print(f"SoftmaxModule (with mask) - Shape of results: {probabilities_with_mask.shape}") # 预期: (batch_size, num_total_jobs)
    print(f"SoftmaxModule (with mask) - Wyniki (batch 0): {probabilities_with_mask[0]}")
    # 预期: batch 0中，工件1和3的概率应接近0
    print(f"SoftmaxModule (with mask) - Sum of probabilities (batch 0): {probabilities_with_mask[0].sum()}") # 预期: 接近 1.0
    print(f"SoftmaxModule (with mask) - Wyniki (batch 1): {probabilities_with_mask[1]}")
    # 预期: batch 1中，工件0, 2, 4的概率应接近0
    print(f"SoftmaxModule (with mask) - Sum of probabilities (batch 1): {probabilities_with_mask[1].sum()}") # 预期: 接近 1.0