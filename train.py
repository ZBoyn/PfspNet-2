import torch
import torch.nn as nn
import torch.nn.functional as F
from PFSPNet import PFSPNet
from critic_network import CriticNetwork
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from calc.calculate_objectives import calculate_objectives_pytorch
from calc.pareto_archive import ParetoArchive

# ==============================================================================
# MODIFIED train_one_batch function
# 关键修改：使用新的目标函数计算 Cmax，并传入更复杂的问题参数
# ==============================================================================
def train_one_batch(
    actor_model, critic_model, optimizer_actor, optimizer_critic,
    batch_instance_features,
    batch_num_machines_scalar,
    # === 新增：传入权重 ===
    w_cmax, w_tec,
    # ========================
    P_instance, E_instance, R_instance, u_instance, s_instance, f_instance,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    batch_size = batch_instance_features.size(0)
    num_total_jobs = batch_instance_features.size(1)

    actor_model.train()
    critic_model.train()

    batch_instance_features = batch_instance_features.to(device)
    batch_num_machines_scalar = batch_num_machines_scalar.to(device)

    # Actor 执行 (逻辑不变)
    selected_indices, log_probs_distributions, encoded_jobs = actor_model(
        batch_instance_processing_times=batch_instance_features,
        batch_num_machines_scalar=batch_num_machines_scalar,
        max_decode_len=num_total_jobs
    )

    # --- 计算原始目标值 ---
    put_off_matrices_zeros = torch.zeros(batch_size, P_instance.size(1), num_total_jobs, device=device, dtype=torch.long)
    actual_cmax_tensor, actual_tec_tensor = calculate_objectives_pytorch(
        job_sequences=selected_indices,
        put_off_matrices=put_off_matrices_zeros,
        P=P_instance, E=E_instance, R=R_instance, u=u_instance, s=s_instance, f=f_instance,
        device=device
    )
    
    # (重要) 处理无效解
    actual_cmax_tensor[torch.isinf(actual_cmax_tensor)] = 5000.0
    actual_tec_tensor[torch.isinf(actual_tec_tensor)] = 50000.0
    
    
    # --- 处理量纲问题并合并目标 ---
    # 1. 对 Cmax 和 TEC 进行批次归一化 (减去均值，除以标准差)
    #    为防止标准差为0(如果批次中所有值都一样), 在分母上加一个很小的数 epsilon
    # cmax_mean = actual_cmax_tensor.mean()
    # cmax_std = actual_cmax_tensor.std()
    # tec_mean = actual_tec_tensor.mean()
    # tec_std = actual_tec_tensor.std()
    
    # # 只有当批次大小大于1且值不全相同时，std才不为0，才进行归一化
    # if cmax_std > 1e-6:
    #     normalized_cmax = (actual_cmax_tensor - cmax_mean) / (cmax_std + 1e-8)
    # else:
    #     normalized_cmax = torch.zeros_like(actual_cmax_tensor) # 如果值都一样，归一化后为0

    # if tec_std > 1e-6:
    #     normalized_tec = (actual_tec_tensor - tec_mean) / (tec_std + 1e-8)
    # else:
    #     normalized_tec = torch.zeros_like(actual_tec_tensor)
        
    # # 2. 使用预设权重计算加权组合目标
    # #    这个 combined_objective 将作为我们新的奖励信号
    # combined_objective = (w_cmax * normalized_cmax + w_tec * normalized_tec)
    
    # --- 核心修改：用“静态缩放”替代“批次归一化” ---
    # 1. 定义缩放因子 (你可以根据目标的实际范围调整)
    #    目标是让缩放后的 cmax 和 tec 在差不多的数量级上
    CMAX_SCALE_FACTOR = 200.0  # e.g., Cmax=400 -> 1.0
    TEC_SCALE_FACTOR = 8000.0  # e.g., TEC=9000 -> 1.0

    # 2. 计算缩放后的目标值
    #    注意我们这里是直接用原始值相除，目标是降低其数值，使其具有可比性
    #    我们希望模型最小化这两个值
    scaled_cmax = actual_cmax_tensor / CMAX_SCALE_FACTOR
    scaled_tec = actual_tec_tensor / TEC_SCALE_FACTOR

    # 3. 使用预设权重计算组合目标
    #    这个 combined_objective 将作为我们新的奖励信号
    #    注意这里我们是在最小化，所以奖励应该是负的目标值
    combined_objective = - (w_cmax * scaled_cmax + w_tec * scaled_tec)
    
    
    # Critic 执行与更新 (使用组合目标)
    baseline_estimates_b = critic_model(encoded_jobs).squeeze(-1)
    # Critic 应该预测组合目标的期望值，而不是单一目标
    critic_loss = F.mse_loss(baseline_estimates_b, combined_objective.detach())
    optimizer_critic.zero_grad()
    critic_loss.backward(retain_graph=True)
    optimizer_critic.step()

    # Actor 更新 (使用组合目标)
    # 优势现在也基于组合目标
    advantage = combined_objective - baseline_estimates_b.detach()
    gathered_log_probs = torch.gather(log_probs_distributions, 2, selected_indices.unsqueeze(2)).squeeze(2)
    sum_log_probs_for_sequence = gathered_log_probs.sum(dim=1)
    actor_loss = (-advantage * sum_log_probs_for_sequence).mean()
    
    optimizer_actor.zero_grad()
    actor_loss.backward()
    optimizer_actor.step()

    # --- 修改: 返回两个原始目标值以供记录 ---
    # return actor_loss.item(), critic_loss.item(), actual_cmax_tensor.mean().item(), actual_tec_tensor.mean().item()
    
    return actor_loss.item(), critic_loss.item(), actual_cmax_tensor, actual_tec_tensor



if __name__ == '__main__':

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"已为 PyTorch 和 NumPy 设置随机种子: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 定义通用参数 ---
    config_batch_size = 16
    config_num_jobs = 20
    config_num_machines = 5
    config_k_intervals = 10
    config_epochs = 100
    config_batches_per_epoch = 50
    # --- 新增: 定义目标权重 ---
    # 你可以调整这两个权重来探索不同的偏好
    # 例如 w_cmax=0.8, w_tec=0.2 会更侧重于优化 Cmax
    
    # !
    # config_weight_cmax = 0.5
    # config_weight_tec = 0.5
    # print(f"目标权重: Cmax = {config_weight_cmax}, TEC = {config_weight_tec}")


    # --- 新增：定义一个固定的问题实例，用于整个训练过程 ---
    P_instance = torch.rand(config_num_jobs, config_num_machines, device=device) * 20 + 1
    E_instance = torch.rand(config_num_jobs, config_num_machines, device=device) * 10
    R_instance = torch.randint(0, 50, (config_num_jobs,), device=device, dtype=torch.float)
    u_starts = torch.arange(0, 200 * config_k_intervals, 200, device=device, dtype=torch.float)
    s_durations = torch.full((config_k_intervals,), 200, device=device)
    f_factors = torch.randint(1, 6, (config_k_intervals,), device=device, dtype=torch.float)

    # --- 网络参数 (与你原来的代码一致) ---
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

    actor = PFSPNet(encoder_args=encoder_config_args, decoder_args=decoder_config_args).to(device)
    critic = CriticNetwork(**critic_config_args).to(device)
    # !
    opt_actor = optim.Adam(actor.parameters(), lr=1e-5)
    opt_critic = optim.Adam(critic.parameters(), lr=1e-3)

    print("模型和优化器实例化完成。开始训练循环...")
    
    # --- 修改: 增加用于存储 Cmax 数据的列表 ---
    # history_epoch_avg_actor_loss = []
    # history_epoch_avg_critic_loss = []
    # history_epoch_avg_cmax = []
    # history_epoch_best_cmax = []
    # history_epoch_avg_tec = []
    # history_epoch_best_tec = []
    
    # instance_features = torch.stack([P_instance, E_instance], dim=-1)
    # batch_features = instance_features.unsqueeze(0).expand(config_batch_size, -1, -1, -1)
    # dummy_m_scalar = torch.full((config_batch_size, 1), float(config_num_machines), device=device)

    # global_best_cmax = float('inf')
    # global_best_tec = float('inf')
    # global_best_sequence_for_cmax = None
    # global_best_sequence_for_tec = None
    
    
    # # --- 训练循环 ---
    # for epoch in range(config_epochs):
    #     actor.train()
    #     critic.train()
    #     batch_actor_losses, batch_critic_losses, batch_cmaxes, batch_teces = [], [], [], []
    #     epoch_best_cmax_this_epoch = float('inf')
    #     epoch_best_tec_this_epoch = float('inf')

    #     for batch_idx in range(config_batches_per_epoch):
    #         # --- 修改: 将权重传入训练函数 ---
    #         actor_loss_val, critic_loss_val, batch_avg_cmax_val, batch_avg_tec_val = train_one_batch(
    #             actor, critic, opt_actor, opt_critic,
    #             batch_features, dummy_m_scalar,
    #             config_weight_cmax, config_weight_tec, # <--- 传入权重
    #             P_instance, E_instance, R_instance, u_starts, s_durations, f_factors,
    #             device
    #         )
            
    #         batch_actor_losses.append(actor_loss_val)
    #         batch_critic_losses.append(critic_loss_val)
    #         # --- 修改: 同时记录 Cmax 和 TEC ---
    #         batch_cmaxes.append(batch_avg_cmax_val)
    #         batch_teces.append(batch_avg_tec_val)

    #     # --- 修改：在 Epoch 结束时进行评估，找到最佳 Cmax 和 TEC ---
    #     actor.eval()
    #     with torch.no_grad():
    #         # 生成一批解用于评估
    #         eval_batch_size = 128 # 可以使用更大的批次来做评估
    #         eval_features = instance_features.unsqueeze(0).expand(eval_batch_size, -1, -1, -1)
    #         eval_m_scalar = torch.full((eval_batch_size, 1), float(config_num_machines), device=device)
            
    #         current_batch_selected_indices, _, _ = actor(eval_features, eval_m_scalar, max_decode_len=config_num_jobs)
    #         put_off_eval = torch.zeros(eval_batch_size, config_num_machines, config_num_jobs, device=device, dtype=torch.long)
            
    #         cmax_vals, tec_vals = calculate_objectives_pytorch(
    #              current_batch_selected_indices, put_off_eval,
    #              P_instance, E_instance, R_instance, u_starts, s_durations, f_factors, device
    #         )

    #         # 找到这批评估中最好的 Cmax 和 TEC
    #         # min_cmax_in_batch_val = torch.min(cmax_vals[torch.isfinite(cmax_vals)])
    #         # min_tec_in_batch_val = torch.min(tec_vals[torch.isfinite(tec_vals)])
            
    #         # 找到当前批次的最优值和对应的序列
    #         min_cmax_in_batch_val, min_cmax_idx = torch.min(cmax_vals, dim=0)
    #         min_tec_in_batch_val, min_tec_idx = torch.min(tec_vals, dim=0)
            
            
    #         epoch_best_cmax_this_epoch = min_cmax_in_batch_val.item()
    #         epoch_best_tec_this_epoch = min_tec_in_batch_val.item()
            
    #         # --- 核心修改：检查并更新历史最优解 ---
    #         if min_cmax_in_batch_val.item() < global_best_cmax:
    #             global_best_cmax = min_cmax_in_batch_val.item()
    #             # 保存获得这个最优Cmax的解序列
    #             global_best_sequence_for_cmax = current_batch_selected_indices[min_cmax_idx].tolist()
    #             print(f"🎉 New Best Cmax Found: {global_best_cmax:.2f}")

    #         if min_tec_in_batch_val.item() < global_best_tec:
    #             global_best_tec = min_tec_in_batch_val.item()
    #             # 保存获得这个最优TEC的解序列
    #             global_best_sequence_for_tec = min_tec_in_batch_val.tolist()
    #             print(f"🎉 New Best TEC Found: {global_best_tec:.2f}")

    #     # Epoch 总结
    #     avg_epoch_actor_loss = np.mean(batch_actor_losses)
    #     avg_epoch_critic_loss = np.mean(batch_critic_losses)
    #     avg_epoch_cmax = np.mean(batch_cmaxes)
    #     avg_epoch_tec = np.mean(batch_teces)

    #     history_epoch_avg_actor_loss.append(avg_epoch_actor_loss)
    #     history_epoch_avg_critic_loss.append(avg_epoch_critic_loss)
    #     history_epoch_avg_cmax.append(avg_epoch_cmax)
    #     history_epoch_best_cmax.append(epoch_best_cmax_this_epoch)
    #     history_epoch_avg_tec.append(avg_epoch_tec)
    #     history_epoch_best_tec.append(epoch_best_tec_this_epoch)
    
    #     print(f"--- Epoch {epoch+1}/{config_epochs} 总结 ---")
    #     print(f"  损失: Actor={avg_epoch_actor_loss:.4f}, Critic={avg_epoch_critic_loss:.4f}")
    #     print(f"  平均目标值: Cmax={avg_epoch_cmax:.2f}, TEC={avg_epoch_tec:.2f}")
    #     print(f"  本 Epoch 最优: Cmax={epoch_best_cmax_this_epoch:.2f}, TEC={epoch_best_tec_this_epoch:.2f}\n")

    # print("训练完成!")

    # # --- 修改: 绘制 Cmax 和 TEC 的双目标图以及损失图 ---
    # epochs_range = range(1, config_epochs + 1)
    # plt.figure(figsize=(20, 6))

    # # 图1: Cmax 随 Epoch 变化
    # plt.subplot(1, 3, 1)
    # plt.plot(epochs_range, history_epoch_avg_cmax, label='Avg Cmax per Epoch', marker='.', color='royalblue')
    # plt.plot(epochs_range, history_epoch_best_cmax, label='Best Cmax in Epoch', marker='x', linestyle='--', color='deepskyblue')
    # plt.xlabel("Epoch")
    # plt.ylabel("Cmax")
    # plt.title("Cmax Performance over Epochs")
    # plt.legend()
    # plt.grid(True)
    
    # # 图2: TEC 随 Epoch 变化
    # plt.subplot(1, 3, 2)
    # plt.plot(epochs_range, history_epoch_avg_tec, label='Avg TEC per Epoch', marker='.', color='darkorange')
    # plt.plot(epochs_range, history_epoch_best_tec, label='Best TEC in Epoch', marker='x', linestyle='--', color='gold')
    # plt.xlabel("Epoch")
    # plt.ylabel("TEC")
    # plt.title("TEC Performance over Epochs")
    # plt.legend()
    # plt.grid(True)

    # # 图3: 损失函数随 Epoch 变化
    # plt.subplot(1, 3, 3)
    # plt.plot(epochs_range, history_epoch_avg_actor_loss, label='Avg Actor Loss', marker='.', color='red')
    # plt.plot(epochs_range, history_epoch_avg_critic_loss, label='Avg Critic Loss', marker='.', color='green', alpha=0.7)
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Losses per Epoch")
    # plt.legend()
    # plt.grid(True)
    
    # plt.tight_layout()
    # plt.show()
    
    # print("\n--- Training Finished ---")
    # print(f"Global Best Cmax: {global_best_cmax:.2f}")
    # print(f"Found with sequence: {global_best_sequence_for_cmax}")
    # print(f"Global Best TEC: {global_best_tec:.2f}")
    # print(f"Found with sequence: {global_best_sequence_for_tec}")
    
    # 1. 实例化帕累托存档
    pareto_archive = ParetoArchive(capacity=100) # 存档容量可自行调整

    history_epoch_avg_actor_loss = []
    history_epoch_avg_critic_loss = []
    history_archive_size = []
    
    instance_features = torch.stack([P_instance, E_instance], dim=-1)
    batch_features = instance_features.unsqueeze(0).expand(config_batch_size, -1, -1, -1)
    dummy_m_scalar = torch.full((config_batch_size, 1), float(config_num_machines), device=device)

    # --- 新的训练循环 ---
    for epoch in range(config_epochs):
        actor.train()
        critic.train()
        batch_actor_losses, batch_critic_losses = [], []

        for batch_idx in range(config_batches_per_epoch):
            # 2. 动态采样权重
            w_cmax = np.random.rand()
            w_tec = 1.0 - w_cmax
            
            # 3. 执行单次训练并获取整批次的目标值
            actor_loss_val, critic_loss_val, cmax_tensor, tec_tensor = train_one_batch(
                actor, critic, opt_actor, opt_critic,
                batch_features, dummy_m_scalar,
                w_cmax, w_tec, # 传入动态权重
                P_instance, E_instance, R_instance, u_starts, s_durations, f_factors,
                device
            )
            
            batch_actor_losses.append(actor_loss_val)
            batch_critic_losses.append(critic_loss_val)
            
            # 4. 将批次中所有有效解尝试添加到存档
            sequences = actor.last_selected_indices
            for i in range(cmax_tensor.size(0)):
                cmax_val = cmax_tensor[i].item()
                tec_val = tec_tensor[i].item()
                # 确保解是有效的
                if np.isfinite(cmax_val) and np.isfinite(tec_val):
                    pareto_archive.add([cmax_val, tec_val], sequences[i].tolist())

        # --- Epoch 总结 ---
        avg_epoch_actor_loss = np.mean(batch_actor_losses) if batch_actor_losses else 0
        avg_epoch_critic_loss = np.mean(batch_critic_losses) if batch_critic_losses else 0
        
        history_epoch_avg_actor_loss.append(avg_epoch_actor_loss)
        history_epoch_avg_critic_loss.append(avg_epoch_critic_loss)
        history_archive_size.append(len(pareto_archive.solutions))
   
        print(f"--- Epoch {epoch+1}/{config_epochs} 总结 ---")
        print(f"  存档大小 (Archive Size): {len(pareto_archive.solutions)}")
        print(f"  平均损失: Actor={avg_epoch_actor_loss:.4f}, Critic={avg_epoch_critic_loss:.4f}\n")

    print("训练完成!")

    # --- 训练后评估与绘图 ---
    # 最终结果就是帕累托存档
    final_solutions = pareto_archive.solutions
    final_sequences = pareto_archive.sequences

    # 1. 绘制帕累托前沿图
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    if final_solutions:
        front = np.array(final_solutions)
        # 按Cmax排序以获得更美观的连线图
        front = front[front[:, 0].argsort()]
        plt.scatter(front[:, 0], front[:, 1], c='red', zorder=2, label='Pareto Solutions')
        plt.plot(front[:, 0], front[:, 1], '--', c='blue', zorder=1, label='Pareto Front')
    plt.xlabel("Cmax (Makespan)")
    plt.ylabel("TEC (Total Energy Cost)")
    plt.title(f"Final Pareto Front (Size: {len(final_solutions)})")
    plt.legend()
    plt.grid(True)

    # 2. 绘制损失和存档大小变化图
    plt.subplot(1, 2, 2)
    epochs_range = range(1, config_epochs + 1)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    p1, = ax1.plot(epochs_range, history_epoch_avg_actor_loss, 'r-', label='Actor Loss')
    p2, = ax1.plot(epochs_range, history_epoch_avg_critic_loss, 'g-', label='Critic Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    p3, = ax2.plot(epochs_range, history_archive_size, 'b-', label='Archive Size')
    ax2.set_ylabel('Archive Size', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    plt.title('Losses and Archive Size over Epochs')
    ax1.legend(handles=[p1, p2, p3], loc='best')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # 3. 打印部分最优解
    print("\n--- 找到的部分帕累托最优解 ---")
    # 为了方便查看，按 Cmax 排序
    sorted_solutions = sorted(zip(final_solutions, final_sequences), key=lambda x: x[0][0])
    for i, (sol, seq) in enumerate(sorted_solutions):
        if i < 10 or i > len(sorted_solutions) - 10: # 只打印前10和后10个
            print(f"  解 {i+1}: Cmax={sol[0]:.2f}, TEC={sol[1]:.2f}") # , Sequence={seq[:10]}...")
