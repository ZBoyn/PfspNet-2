import torch
import torch.nn.functional as F
import numpy as np
import torch
from PFSPNet import PFSPNet
from critic_network import CriticNetwork
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from calculate_objectives import calculate_objectives_pytorch
from pareto_archive import ParetoArchive
from heuristicScheduler import HeuristicScheduler

def train_one_batch_with_heuristic_reward(
    actor_model, critic_model, optimizer_actor, optimizer_critic,
    batch_instance_features,
    batch_num_machines_scalar,
    # ! 高质量启发式调度器实例
    heuristic_scheduler,
    # ---
    w_cmax, w_tec,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    使用高质量启发式调度器返回的目标值作为奖励来训练一个批次。
    """
    batch_size = batch_instance_features.size(0)
    num_total_jobs = batch_instance_features.size(1)

    actor_model.train()
    critic_model.train()

    batch_instance_features = batch_instance_features.to(device)
    batch_num_machines_scalar = batch_num_machines_scalar.to(device)

    # Actor 执行，得到一批序列
    selected_indices, log_probs_distributions, encoded_jobs = actor_model(
        batch_instance_processing_times=batch_instance_features,
        batch_num_machines_scalar=batch_num_machines_scalar,
        max_decode_len=num_total_jobs
    )
    actor_model.last_selected_indices = selected_indices # 暂存序列

    # 1. 将 Actor 生成的序列从 GPU 转到 CPU (List of Lists 格式)
    sequences_to_evaluate = selected_indices.cpu().tolist()

    # 2. 调用启发式调度器进行评估
    #    这将是整个训练步骤中最耗时的部分
    heuristic_cmax_list, heuristic_tec_list, _ = heuristic_scheduler.evaluate(sequences_to_evaluate)

    # 3. 将返回的 Python list 结果转回 GPU 上的 Tensor
    actual_cmax_tensor = torch.tensor(heuristic_cmax_list, device=device, dtype=torch.float)
    actual_tec_tensor = torch.tensor(heuristic_tec_list, device=device, dtype=torch.float)
    
    actual_cmax_tensor[torch.isinf(actual_cmax_tensor)] = 5000.0  # 使用惩罚值
    actual_tec_tensor[torch.isinf(actual_tec_tensor)] = 50000.0 # 使用惩罚值

    
    # 使用"静态缩放"处理量纲问s题
    CMAX_SCALE_FACTOR = 200.0
    TEC_SCALE_FACTOR = 8000.0
    scaled_cmax = actual_cmax_tensor / CMAX_SCALE_FACTOR
    scaled_tec = actual_tec_tensor / TEC_SCALE_FACTOR

    # combined_objective = - (w_cmax * scaled_cmax + w_tec * scaled_tec)
    
    # Pareto 方法计算目标值
    pareto_objective = - torch.max(w_cmax * scaled_cmax, w_tec * scaled_tec)
    
    baseline_estimates_b = critic_model(encoded_jobs).squeeze(-1)
    critic_loss = F.mse_loss(baseline_estimates_b, pareto_objective.detach())
    optimizer_critic.zero_grad()
    critic_loss.backward(retain_graph=True)
    optimizer_critic.step()

    # Actor 更新 (基于高质量奖励)
    advantage = pareto_objective - baseline_estimates_b.detach()
    gathered_log_probs = torch.gather(log_probs_distributions, 2, selected_indices.unsqueeze(2)).squeeze(2)
    sum_log_probs_for_sequence = gathered_log_probs.sum(dim=1)
    actor_loss = (-advantage * sum_log_probs_for_sequence).mean()
    
    optimizer_actor.zero_grad()
    actor_loss.backward()
    optimizer_actor.step()

    # 返回高质量的目标值，用于记录和更新帕累托存档
    return actor_loss.item(), critic_loss.item(), actual_cmax_tensor, actual_tec_tensor

def evaluate_model_with_weights(actor_model, heuristic_scheduler, batch_features, dummy_m_scalar, 
                               w_cmax, w_tec, device, num_eval_batches=10):
    """
    使用指定权重评估模型，返回帕累托前沿
    """
    actor_model.eval()
    pareto_archive = ParetoArchive(capacity=100)
    
    with torch.no_grad():
        for _ in range(num_eval_batches):
            # 生成序列
            selected_indices, _, _ = actor_model(
                batch_instance_processing_times=batch_features,
                batch_num_machines_scalar=dummy_m_scalar,
                max_decode_len=batch_features.size(1)
            )
            
            # 评估序列
            sequences_to_evaluate = selected_indices.cpu().tolist()
            heuristic_cmax_list, heuristic_tec_list, _ = heuristic_scheduler.evaluate(sequences_to_evaluate)
            
            # 添加到帕累托存档
            for i in range(len(heuristic_cmax_list)):
                cmax_val = heuristic_cmax_list[i]
                tec_val = heuristic_tec_list[i]
                if np.isfinite(cmax_val) and np.isfinite(tec_val):
                    pareto_archive.add([cmax_val, tec_val], selected_indices[i].tolist())
    
    return pareto_archive.solutions

def train_model_with_weights(encoder_config_args, decoder_config_args, critic_config_args,
                            batch_features, dummy_m_scalar, heuristic_scheduler,
                            w_cmax, w_tec, device, config_epochs=50, config_batches_per_epoch=50):
    """
    使用指定权重训练一个完整的模型
    """
    # 创建新的模型实例
    actor = PFSPNet(encoder_args=encoder_config_args, decoder_args=decoder_config_args).to(device)
    critic = CriticNetwork(**critic_config_args).to(device)
    opt_actor = optim.Adam(actor.parameters(), lr=1e-5)
    opt_critic = optim.Adam(critic.parameters(), lr=1e-3)
    
    # 实例化帕累托存档
    pareto_archive = ParetoArchive(capacity=100)
    
    print(f"开始训练权重 w_cmax={w_cmax:.1f}, w_tec={w_tec:.1f} 的模型...")
    
    for epoch in range(config_epochs):
        actor.train()
        critic.train()
        batch_actor_losses, batch_critic_losses = [], []

        for batch_idx in range(config_batches_per_epoch):
            actor_loss_val, critic_loss_val, cmax_tensor, tec_tensor = train_one_batch_with_heuristic_reward(
                actor, critic, opt_actor, opt_critic,
                batch_features, dummy_m_scalar,
                heuristic_scheduler,
                w_cmax, w_tec,
                device
            )
            
            batch_actor_losses.append(actor_loss_val)
            batch_critic_losses.append(critic_loss_val)
            
            # 将批次中所有解尝试添加到存档
            sequences = actor.last_selected_indices
            for i in range(cmax_tensor.size(0)):
                cmax_val = cmax_tensor[i].item()
                tec_val = tec_tensor[i].item()
                if np.isfinite(cmax_val) and np.isfinite(tec_val):
                    pareto_archive.add([cmax_val, tec_val], sequences[i].tolist())

        if (epoch + 1) % 10 == 0:  # 每10个epoch打印一次
            avg_epoch_actor_loss = np.mean(batch_actor_losses) if batch_actor_losses else 0
            avg_epoch_critic_loss = np.mean(batch_critic_losses) if batch_critic_losses else 0
            print(f"  Epoch {epoch+1}/{config_epochs}: Actor Loss={avg_epoch_actor_loss:.4f}, Critic Loss={avg_epoch_critic_loss:.4f}, Archive Size={len(pareto_archive.solutions)}")
    
    print(f"权重 w_cmax={w_cmax:.1f}, w_tec={w_tec:.1f} 的模型训练完成!")
    return actor, pareto_archive.solutions

if __name__ == '__main__':

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"已为 PyTorch 和 NumPy 设置随机种子: {seed}")

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    config_batch_size = 16
    config_num_jobs = 20
    config_num_machines = 5
    config_k_intervals = 10
    config_epochs = 50
    config_batches_per_epoch = 50
    
    P_instance = torch.rand(config_num_jobs, config_num_machines, device=device) * 20 + 1
    E_instance = torch.rand(config_num_jobs, config_num_machines, device=device) * 10
    R_instance = torch.randint(0, 50, (config_num_jobs,), device=device, dtype=torch.float)
    u_starts = torch.arange(0, 200 * config_k_intervals, 200, device=device, dtype=torch.float)
    s_durations = torch.full((config_k_intervals,), 200, device=device)
    f_factors = torch.randint(1, 6, (config_k_intervals,), device=device, dtype=torch.float)

    NEW_SCALAR_INPUT_DIM = 2 
    enc_part1_args = {'scalar_input_dim': NEW_SCALAR_INPUT_DIM, 'embedding_dim': 32, 'hidden_dim': 64, 'rnn_type': 'RNN', 'num_rnn_layers':1}
    enc_part2_args = {'p_vector_dim': enc_part1_args['hidden_dim'], 'm_embedding_dim': 16, 'output_dim': 48}
    ENC_OUT_CHANNELS = 96
    enc_part3_args = {'p_tilde_dim': enc_part2_args['output_dim'], 'conv_out_channels': ENC_OUT_CHANNELS, 'conv_kernel_size': 3, 'conv_padding': 'same'}
    encoder_config_args = {'part1_args': enc_part1_args, 'part2_args': enc_part2_args, 'part3_args': enc_part3_args}
    dec_step1_pt_enc_args = {'scalar_input_dim': NEW_SCALAR_INPUT_DIM, 'embedding_dim': 30, 'hidden_dim': 50, 'rnn_type': 'RNN', 'num_rnn_layers': 1}
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

    # 实例化启发式调度器
    heuristic_scheduler = HeuristicScheduler(
        M=config_num_machines, N=config_num_jobs, K=config_k_intervals,
        P=P_instance.cpu().numpy().T, E=E_instance.cpu().numpy().T,
        R=R_instance.cpu().numpy(),
        U=torch.cat([torch.tensor([0]), u_starts + s_durations]).cpu().numpy(),
        S=torch.cat([torch.tensor([0]), u_starts]).cpu().numpy(),
        W=f_factors.cpu().numpy()
    )
    print("高质量启发式调度器已实例化。")

    instance_features = torch.stack([P_instance, E_instance], dim=-1)
    batch_features = instance_features.unsqueeze(0).expand(config_batch_size, -1, -1, -1)
    dummy_m_scalar = torch.full((config_batch_size, 1), float(config_num_machines), device=device)

    # ======================================================================
    # --- 为每组权重训练独立的模型 ---
    # ======================================================================
    
    print("开始为不同权重组合训练独立的模型...")
    
    # 生成权重组合 (w_cmax从0到1，间隔0.1)
    weight_combinations = []
    for w_cmax in np.arange(0, 1.1, 0.1):
        w_tec = 1.0 - w_cmax
        weight_combinations.append((w_cmax, w_tec))
    
    all_trained_models = []
    all_pareto_fronts = []
    all_weights = []
    
    for i, (w_cmax, w_tec) in enumerate(weight_combinations):
        print(f"\n=== 训练权重组合 {i+1}/{len(weight_combinations)}: w_cmax={w_cmax:.1f}, w_tec={w_tec:.1f} ===")
        
        # 为当前权重训练一个完整的模型
        trained_actor, pareto_front = train_model_with_weights(
            encoder_config_args, decoder_config_args, critic_config_args,
            batch_features, dummy_m_scalar, heuristic_scheduler,
            w_cmax, w_tec, device, config_epochs, config_batches_per_epoch
        )
        
        all_trained_models.append(trained_actor)
        all_pareto_fronts.append(pareto_front)
        all_weights.append((w_cmax, w_tec))
        
        print(f"  权重 w_cmax={w_cmax:.1f}, w_tec={w_tec:.1f} 训练完成，找到 {len(pareto_front)} 个帕累托解")
    
    print(f"\n所有 {len(weight_combinations)} 个权重组合的模型训练完成!")
    
    # ======================================================================
    # --- 绘制所有帕累托前沿 ---
    # ======================================================================
    
    plt.figure(figsize=(20, 8))
    
    # 1. 绘制所有帕累托前沿
    plt.subplot(1, 2, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_pareto_fronts)))
    
    for i, (pareto_front, (w_cmax, w_tec)) in enumerate(zip(all_pareto_fronts, all_weights)):
        if pareto_front:
            front = np.array(pareto_front)
            # 按Cmax排序以获得更美观的连线图
            front = front[front[:, 0].argsort()]
            plt.scatter(front[:, 0], front[:, 1], c=[colors[i]], s=30, alpha=0.7, 
                       label=f'w_cmax={w_cmax:.1f}, w_tec={w_tec:.1f}')
            plt.plot(front[:, 0], front[:, 1], '--', c=colors[i], alpha=0.5, linewidth=1)
    
    plt.xlabel("Cmax (Makespan)")
    plt.ylabel("TEC (Total Energy Cost)")
    plt.title(f"不同权重训练的模型帕累托前沿 (共{len(weight_combinations)}组)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 2. 绘制权重分布图
    plt.subplot(1, 2, 2)
    w_cmax_values = [w[0] for w in all_weights]
    w_tec_values = [w[1] for w in all_weights]
    pareto_sizes = [len(front) if front else 0 for front in all_pareto_fronts]
    
    plt.scatter(w_cmax_values, pareto_sizes, c=colors, s=100, alpha=0.8)
    plt.xlabel("w_cmax 权重")
    plt.ylabel("帕累托解数量")
    plt.title("不同权重对应的帕累托解数量")
    plt.grid(True, alpha=0.3)
    
    # 添加权重标签
    for i, (w_cmax, size) in enumerate(zip(w_cmax_values, pareto_sizes)):
        plt.annotate(f'{w_cmax:.1f}', (w_cmax, size), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # 3. 单独绘制帕累托前沿对比图
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_pareto_fronts)))
    
    for i, (pareto_front, (w_cmax, w_tec)) in enumerate(zip(all_pareto_fronts, all_weights)):
        if pareto_front:
            front = np.array(pareto_front)
            front = front[front[:, 0].argsort()]
            plt.scatter(front[:, 0], front[:, 1], c=[colors[i]], s=50, alpha=0.8, 
                       label=f'w_cmax={w_cmax:.1f}, w_tec={w_tec:.1f}')
            plt.plot(front[:, 0], front[:, 1], '--', c=colors[i], alpha=0.6, linewidth=2)
    
    plt.xlabel("Cmax (Makespan)")
    plt.ylabel("TEC (Total Energy Cost)")
    plt.title("不同权重训练的模型帕累托前沿对比")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 4. 打印统计信息
    print("\n--- 不同权重训练的模型统计信息 ---")
    for i, (pareto_front, (w_cmax, w_tec)) in enumerate(zip(all_pareto_fronts, all_weights)):
        if pareto_front:
            front = np.array(pareto_front)
            cmax_range = (front[:, 0].min(), front[:, 0].max())
            tec_range = (front[:, 1].min(), front[:, 1].max())
            print(f"权重 w_cmax={w_cmax:.1f}, w_tec={w_tec:.1f}:")
            print(f"  解数量: {len(pareto_front)}")
            print(f"  Cmax范围: [{cmax_range[0]:.2f}, {cmax_range[1]:.2f}]")
            print(f"  TEC范围: [{tec_range[0]:.2f}, {tec_range[1]:.2f}]")
        else:
            print(f"权重 w_cmax={w_cmax:.1f}, w_tec={w_tec:.1f}: 无有效解")
        print()

    # 5. 分析权重对性能的影响
    print("\n--- 权重对性能的影响分析 ---")
    valid_results = [(w, len(front)) for w, front in zip(all_weights, all_pareto_fronts) if front]
    if valid_results:
        weights, sizes = zip(*valid_results)
        w_cmax_list = [w[0] for w in weights]
        w_tec_list = [w[1] for w in weights]
        
        print(f"平均帕累托解数量: {np.mean(sizes):.2f}")
        print(f"最大帕累托解数量: {max(sizes)} (权重: w_cmax={w_cmax_list[sizes.index(max(sizes))]:.1f})")
        print(f"最小帕累托解数量: {min(sizes)} (权重: w_cmax={w_cmax_list[sizes.index(min(sizes))]:.1f})")
        
        # 计算权重与解数量的相关性
        correlation = np.corrcoef(w_cmax_list, sizes)[0, 1]
        print(f"w_cmax权重与解数量的相关系数: {correlation:.3f}")
