import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['SimHei']
from calc.calculateWithDumny import CalculateWithDummy, virtual_due_date, ProblemParameters

from PFSPNet import PFSPNet
from critic_network import CriticNetwork

def pretrain_due_date_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    P, E, R, U, S, W = config['P'], config['E'], config['R'], config['U'], config['S'], config['W']
    M, N, K = config['M'], config['N'], config['K']

    D = virtual_due_date(R, P, slack_factor=1.5)
    problem_params = ProblemParameters(M, N, K, P, E, D, R, U, S, W)
    calculator = CalculateWithDummy(problem_params)

    # 初始化 Actor和 Critic 网络
    num_total_items = N + K - 1  # 虚拟节点数量和真实工件数量之和

    SCALAR_INPUT_DIM = 2
    enc_part1_args = {'scalar_input_dim': SCALAR_INPUT_DIM, 'embedding_dim': 32, 'hidden_dim': 64, 'rnn_type': 'RNN', 'num_rnn_layers':1}
    enc_part2_args = {'p_vector_dim': enc_part1_args['hidden_dim'], 'm_embedding_dim': 16, 'output_dim': 48}
    ENC_OUT_CHANNELS = 96
    enc_part3_args = {'p_tilde_dim': enc_part2_args['output_dim'], 'conv_out_channels': ENC_OUT_CHANNELS, 'conv_kernel_size': 3, 'conv_padding': 'same'}
    encoder_config_args = {'part1_args': enc_part1_args, 'part2_args': enc_part2_args, 'part3_args': enc_part3_args}
    
    dec_step1_pt_enc_args = {'scalar_input_dim': SCALAR_INPUT_DIM, 'embedding_dim': 30, 'hidden_dim': 50, 'rnn_type': 'RNN', 'num_rnn_layers': 1}
    decoder_config_args = {
        'step1_pt_encoder_args': dec_step1_pt_enc_args,
        'step1_m_embedding_dim': 20, 'step1_di_output_dim': 40, 'step1_fc_hidden_dims': [35],
        'step2_rnn2_hidden_dim': 70, 'step2_rnn_type': 'LSTM', 'step2_num_rnn_layers': 1,
        'attention_job_encoding_dim': ENC_OUT_CHANNELS,
        'attention_hidden_dim': 55,
        'ptr_dim': 25
    }
    critic_config_args = {
        'encoder_output_dim': ENC_OUT_CHANNELS,
        'conv_channels_list': [64, 128, 64],
        'conv_kernel_sizes_list': [3, 3, 3],
        'final_fc_hidden_dims': [32]
    }

    actor = PFSPNet(encoder_args=encoder_config_args, decoder_args=decoder_config_args).to(device)
    critic = CriticNetwork(**critic_config_args).to(device)

    opt_actor = optim.Adam(actor.parameters(), lr=config['lr_actor'])
    opt_critic = optim.Adam(critic.parameters(), lr=config['lr_critic'])

    # 真实工件特征
    real_job_features = torch.tensor(np.stack([P, E], axis=-1), dtype=torch.float32)
    # 虚拟节点特征
    dummy_job_features = torch.zeros((K - 1, M, 2), dtype=torch.float32)
    # 所有物品特征
    all_item_features = torch.cat([real_job_features, dummy_job_features], dim=0).to(device)    

    # 扩展成批次
    batch_features = all_item_features.unsqueeze(0).expand(config['batch_size'], -1, -1, -1)
    batch_m_scalar = torch.full((config['batch_size'], 1), float(M), device=device)

    history_total_tardiness = []
    print("\n开始预训练，目标：最小化总延期...")
    
    for epoch in range(config['epochs']):
        actor.train()
        critic.train()

        # Actor 生成一批序列
        # 注意：max_decode_len 现在是总物品数
        sequences, log_probs_distributions, encoded_jobs = actor(
            batch_instance_processing_times=batch_features,
            batch_num_machines_scalar=batch_m_scalar,
            max_decode_len=num_total_items
        )

        # 【核心】为批次中的每个序列计算奖励
        rewards = []
        batch_tardiness = []
        for seq in sequences:
            _cmax, _tec, total_tardiness = calculator.calculate(seq.cpu().numpy())
            
            # 如果序列无效，给一个大的惩罚
            if total_tardiness == float('inf'):
                reward = -1e9 # 大负数作为惩罚
            else:
                # 奖励 = -总延期。我们希望延期越小越好，对应奖励越大越好
                reward = -total_tardiness
            
            rewards.append(reward)
            batch_tardiness.append(total_tardiness if total_tardiness != float('inf') else 1e9)
        
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        
        # Critic 学习评估状态的价值 (这里的价值就是我们定义的奖励)
        baseline_estimates = critic(encoded_jobs).squeeze(-1)
        critic_loss = F.mse_loss(baseline_estimates, rewards)
        
        opt_critic.zero_grad()
        critic_loss.backward(retain_graph=True) # retain_graph 以便 actor 使用 encoded_jobs
        opt_critic.step()
        
        # Actor 根据 Critic 的评估进行学习
        advantage = (rewards - baseline_estimates.detach())
        
        # 收集所选动作的对数概率
        gathered_log_probs = torch.gather(log_probs_distributions, 2, sequences.unsqueeze(2)).squeeze(2)
        sum_log_probs = gathered_log_probs.sum(dim=1)
        
        actor_loss = (-advantage * sum_log_probs).mean()
        
        opt_actor.zero_grad()
        actor_loss.backward()
        opt_actor.step()
        
        avg_tardiness = np.mean(batch_tardiness)
        history_total_tardiness.append(avg_tardiness)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{config['epochs']}] | 平均延期: {avg_tardiness:.2f} | Actor Loss: {actor_loss.item():.4f} | Critic Loss: {critic_loss.item():.4f}")

    # --- 5. 结束训练，保存模型和绘图 ---
    print("\n预训练完成！")
    torch.save(actor.state_dict(), "duedate_pretrained_actor.pth")
    torch.save(critic.state_dict(), "duedate_pretrained_critic.pth")
    print("预训练模型已保存为 'duedate_pretrained_actor.pth' 和 'duedate_pretrained_critic.pth'")
    
    plt.figure(figsize=(10, 5))
    plt.plot(history_total_tardiness)
    plt.title("预训练过程中的平均总延期变化")
    plt.xlabel("Epoch")
    plt.ylabel("平均总延期 (Total Tardiness)")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # --- 定义一个具体的问题和训练配置 ---
    config = {
        # 问题参数
        "M": 3,
        "N": 10, # 增加工件数量以更好地体现学习效果
        "K": 4,  # 增加时段数量
        "P": np.random.randint(5, 20, size=(10, 3)),
        "E": np.random.randint(2, 5, size=(10, 3)),
        "R": np.random.randint(0, 40, size=10),
        "U": np.array([0, 50, 100, 150]),
        "S": np.array([50, 100, 150, 300]),
        "W": np.array([1.0, 1.5, 0.8, 1.2]),
        
        # 训练参数
        "slack_factor": 2.0,
        "batch_size": 32, # 批次大小
        "epochs": 2000,
        "lr_actor": 3e-5,
        "lr_critic": 1e-4,
    }

    pretrain_due_date_model(config)