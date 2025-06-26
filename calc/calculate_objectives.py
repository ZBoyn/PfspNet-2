import torch
import bisect


def calculate_objectives_pytorch(
    job_sequences,       # (B, N)
    put_off_matrices,    # (B, M, N)
    P,                   # (N, M)
    E,                   # (N, M)
    R,                   # (N,)
    u,                   # (K,)
    s,                   # (K,)
    f,                   # (K,)
    device,
    sanity_threshold=5000.0
):
    P, E, R, u, s, f = P.to(device), E.to(device), R.to(device), u.to(device), s.to(device), f.to(device)
    batch_size, n_jobs = job_sequences.shape
    n_machines = P.shape[1]
    k_intervals = u.shape[0]
    u_end = u + s
    batch_cmax, batch_tec = torch.zeros(batch_size, device=device), torch.zeros(batch_size, device=device)

    for b in range(batch_size):
        job_seq, put_off = job_sequences[b], put_off_matrices[b]
        completion_times = torch.zeros((n_jobs, n_machines), device=device)
        mach_free_times = torch.zeros(n_machines, device=device)
        total_energy_cost = 0.0
        is_invalid = False

        for i in range(n_jobs):
            if is_invalid: break
            job_idx = job_seq[i].item()
            for m in range(n_machines):
                p_jm, e_jm = P[job_idx, m], E[job_idx, m]
                release_time = R[job_idx]
                prev_mach_completion = completion_times[job_idx, m-1] if m > 0 else torch.tensor(0.0, device=device)
                est = torch.max(torch.stack([mach_free_times[m], release_time, prev_mach_completion]))

                if est.item() > sanity_threshold:
                    is_invalid = True; break

                current_period_idx = torch.searchsorted(u, est, right=True) - 1
                delay = put_off[m, i].long()
                delayed_period_idx = current_period_idx + delay
                if delayed_period_idx.item() >= k_intervals: is_invalid = True; break
                est = torch.max(est, u[delayed_period_idx])
                
                start_time, found_slot = est, False
                search_idx = torch.searchsorted(u, start_time, right=True) - 1
                if search_idx < 0: search_idx = 0
                while not found_slot:
                    if search_idx >= k_intervals: start_time = float('inf'); break
                    potential_start = torch.max(start_time, u[search_idx])
                    if potential_start + p_jm <= u_end[search_idx]:
                        start_time, found_slot = potential_start, True
                    else:
                        search_idx += 1
                        if search_idx < k_intervals: start_time = u[search_idx]
                        else: start_time = float('inf'); break
                
                if start_time == float('inf'): is_invalid = True; break
                
                end_time = start_time + p_jm
                completion_times[job_idx, m], mach_free_times[m] = end_time, end_time
                final_period_idx = torch.searchsorted(u, start_time, right=True) - 1
                total_energy_cost += p_jm * e_jm * f[final_period_idx]

        if is_invalid:
            batch_cmax[b], batch_tec[b] = float('inf'), float('inf')
        else:
            final_cmax = torch.max(completion_times[:, -1])
            if final_cmax.item() > sanity_threshold:
                 batch_cmax[b], batch_tec[b] = float('inf'), float('inf')
            else:
                 batch_cmax[b], batch_tec[b] = final_cmax, total_energy_cost
    
    return batch_cmax, batch_tec

if __name__ == "__main__":
    job_sequences = torch.tensor([[2, 0, 3, 4, 5, 6, 1], [2, 6, 5, 4, 0, 3, 1]])  # (batch_size, num_jobs)
    put_off_matrices = torch.tensor([[[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]]])  # (batch_size, num_machines, num_jobs)
    P = torch.tensor([[4, 6, 4], [6, 1, 1], [5, 3, 2], [2, 5, 2], [5, 2, 3], [6, 7, 5], [9, 10, 4]])
    E = torch.tensor([[7, 4, 4], [3, 3, 3], [4, 3, 6], [2, 6, 3], [7, 2, 6], [2, 5, 3], [8, 2, 5]])  # (num_jobs, num_machines)
    R = torch.tensor([14, 85, 0, 16, 20, 40, 50])  # (num_jobs,)
    u = torch.tensor([0, 30, 60, 80, 100])  # (k_intervals,)
    s = torch.tensor([30, 30, 20, 20, 20])  # (k_intervals,)
    f = torch.tensor([2, 5, 3, 4, 1])  # (k_intervals,)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cmax, tec = calculate_objectives_pytorch(
        job_sequences, put_off_matrices, P, E, R, u, s, f, device
    )
    print(f"Cmax: {cmax}")
    print(f"Total Energy Cost (TEC): {tec}")    