from datastreams import generate_water_flow_data
from models import *
import numpy as np
import pandas as pd
from driftdetector import DriftDetector
from tqdm import tqdm

def make_dataset(data, seq_len):
    time_series = []
    for i in range(len(data) - seq_len + 1):
        time_series.append(data[i:i + seq_len])
    return np.array(time_series)[:-1], data[seq_len:]

def increase_lr_only(optimizer, drifted, decay_state, factor=2.0, max_lr=0.01):
    if drifted:
        for i, pg in enumerate(optimizer.param_groups):
            base_lr = decay_state['base_lrs'][i]
            pg['lr'] = min(base_lr * factor, max_lr)

def increase_and_decay_lr(optimizer, drifted, decay_state, factor=2.0, decay_steps=10):
    base_lrs = decay_state['base_lrs']
    
    if drifted:
        for i, pg in enumerate(optimizer.param_groups):
            pg['lr'] = base_lrs[i] * factor
        decay_state['current_decay_step'] = decay_steps

    elif decay_state['current_decay_step'] > 0:
        decay_state['current_decay_step'] -= 1
        for i, pg in enumerate(optimizer.param_groups):
            boosted_lr = base_lrs[i] * factor
            ratio = decay_state['current_decay_step'] / decay_steps
            pg['lr'] = base_lrs[i] + (boosted_lr - base_lrs[i]) * ratio

def train_model_mlp(model, detector, x_all, y_all, optimizer, loss_fn, device, drift_action):

    model.train()
    all_loss = []

    # For drift action state
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    decay_state = {
        'base_lrs': base_lrs,
        'current_decay_step': 0
    }

    step = 0
    mae_values = []
    mae_values_full = []

    for x, y in zip(x_all, y_all):
        step += 1
        x_ten = torch.tensor(x, dtype=torch.float32).view(1, -1).to(device)
        y_ten = torch.tensor(y, dtype=torch.float32).view(1, -1).to(device)
        # Training step
        optimizer.zero_grad()
        pred = model(x_ten)
        loss = loss_fn(pred, y_ten)
        loss.backward()
        optimizer.step()
        all_loss.append(loss.item())

        # Drift handling
        if detector.update(loss.item()):
            drift_action(optimizer, drifted=True, decay_state=decay_state)

        drift_action(optimizer, drifted=False, decay_state=decay_state)
        
        mae_full = torch.abs(pred - y_ten).mean().item()
        mae_values_full.append(mae_full)

        # Post-warmup MAE tracking
        if step >= 200:
            mae_values.append(mae_full)

    avg_loss = sum(all_loss) / len(all_loss) if all_loss else 0.0
    avg_mae_post_warmup = sum(mae_values) / len(mae_values) if mae_values else 0.0
    avg_mae_full = sum(mae_values_full) / len(mae_values_full) if mae_values_full else 0.0


    return model, avg_loss, avg_mae_post_warmup, avg_mae_full

def train_model(model, detector, x_all, y_all, seq_len, optimizer, loss_fn, device, drift_action):

    model.train()
    all_loss = []

    # For drift action state
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    decay_state = {
        'base_lrs': base_lrs,
        'current_decay_step': 0
    }

    step = 0
    mae_values = []
    mae_values_full = []

    for x, y in zip(x_all, y_all):
        step += 1
        x_ten = torch.tensor(x, dtype=torch.float32).view(1, seq_len, -1).to(device)
        y_ten = torch.tensor(y, dtype=torch.float32).view(1, -1).to(device)
        # Training step
        optimizer.zero_grad()
        pred = model(x_ten)
        loss = loss_fn(pred, y_ten)
        loss.backward()
        optimizer.step()
        all_loss.append(loss.item())

        # Drift handling
        if detector.update(loss.item()):
            drift_action(optimizer, drifted=True, decay_state=decay_state)

        drift_action(optimizer, drifted=False, decay_state=decay_state)
        
        mae_full = torch.abs(pred - y_ten).mean().item()
        mae_values_full.append(mae_full)

        # Post-warmup MAE tracking
        if step >= 200:
            mae_values.append(mae_full)

    avg_loss = sum(all_loss) / len(all_loss) if all_loss else 0.0
    avg_mae_post_warmup = sum(mae_values) / len(mae_values) if mae_values else 0.0
    avg_mae_full = sum(mae_values_full) / len(mae_values_full) if mae_values_full else 0.0


    return model, avg_loss, avg_mae_post_warmup, avg_mae_full

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector_types = ["ADWIN", "PageHinkley"]
    drift_actions = [increase_lr_only, increase_and_decay_lr]
    seq_lens = [16, 32, 64]
    hidden_sizess = [[16], [32, 32], [64, 64]]
    data = generate_water_flow_data()
    for detector_type in detector_types:
        detector = DriftDetector(method=detector_type)
        for drift_action in drift_actions:
            for seq_len in seq_lens:
                for hidden_sizes in hidden_sizess:
                    x_all, y_all = make_dataset(data, seq_len)
                    model = TimeSeriesMLP(
                        input_size=seq_len,
                        hidden_sizes=hidden_sizes,
                        output_size=1,
                    )
                    model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    loss_fn = nn.MSELoss()
                    model, loss, mae_full, mae_warmup = train_model_mlp(
                        model,
                        detector,
                        x_all,
                        y_all,
                        optimizer,
                        loss_fn,
                        device=device,
                        drift_action=drift_action
                    )
                    # save stream, n_dim, n_points, detector_type, seq_len, hidden_sizes, train_loss, test_loss to CVS file
                    with open("wf_mlp.csv", "a") as f:
                        f.write(
                            f"{generate_water_flow_data.__name__};1;{len(data)};{detector_type};{drift_action.__name__};{seq_len};{hidden_sizes};{loss};{mae_full};{mae_warmup}\n"
                        )

    hidden_sizess = [16, 32, 64]
    data = generate_water_flow_data()
    for detector_type in detector_types:
        detector = DriftDetector(method=detector_type)
        for drift_action in drift_actions:
            for seq_len in seq_lens:
                for hidden_sizes in hidden_sizess:
                    x_all, y_all = make_dataset(data, seq_len)
                    model = RNNModel(
                        input_size=1,
                        hidden_size=hidden_sizes,
                        output_size=1,
                        rnn_type='rnn'
                    )
                    model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    loss_fn = nn.MSELoss()
                    model, loss, mae_full, mae_warmup = train_model(
                        model,
                        detector,
                        x_all,
                        y_all,
                        optimizer,
                        loss_fn,
                        device=device,
                        drift_action=drift_action
                    )
                    # save stream, n_dim, n_points, detector_type, seq_len, hidden_sizes, train_loss, test_loss to CVS file
                    with open("wf_rnn.csv", "a") as f:
                        f.write(
                            f"{generate_water_flow_data.__name__};1;{len(data)};{detector_type};{drift_action.__name__};{seq_len};{hidden_sizes};{loss};{mae_full};{mae_warmup}\n"
                        )

    data = generate_water_flow_data()
    for detector_type in detector_types:
        detector = DriftDetector(method=detector_type)
        for drift_action in drift_actions:
            for seq_len in seq_lens:
                for hidden_sizes in hidden_sizess:
                    x_all, y_all = make_dataset(data, seq_len)
                    model = RNNModel(
                        input_size=1,
                        hidden_size=hidden_sizes,
                        output_size=1,
                        rnn_type='lstm'
                    )
                    model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    loss_fn = nn.MSELoss()
                    model, loss, mae_full, mae_warmup = train_model(
                        model,
                        detector,
                        x_all,
                        y_all,
                        optimizer,
                        loss_fn,
                        device=device,
                        drift_action=drift_action
                    )
                    # save stream, n_dim, n_points, detector_type, seq_len, hidden_sizes, train_loss, test_loss to CVS file
                    with open("wf_lstm.csv", "a") as f:
                        f.write(
                            f"{generate_water_flow_data.__name__};1;{len(data)};{detector_type};{drift_action.__name__};{seq_len};{hidden_sizes};{loss};{mae_full};{mae_warmup}\n"
                        )

    d_models = [16, 32, 64]
    data = generate_water_flow_data()
    for detector_type in detector_types:
        detector = DriftDetector(method=detector_type)
        for drift_action in drift_actions:
            for seq_len in seq_lens:
                for d_model in d_models:
                    x_all, y_all = make_dataset(data, seq_len)
                    model = TimeSeriesTransformer(
                        input_size=1,
                        d_model=d_model,
                        nhead=2,
                        num_layers=2,
                        output_size=1,
                    )
                    model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    loss_fn = nn.MSELoss()
                    model, loss, mae_full, mae_warmup = train_model(
                        model,
                        detector,
                        x_all,
                        y_all,
                        optimizer,
                        loss_fn,
                        device=device,
                        drift_action=drift_action
                    )
                    # save stream, n_dim, n_points, detector_type, seq_len, hidden_sizes, train_loss, test_loss to CVS file
                    with open("wf_transformer.csv", "a") as f:
                        f.write(
                            f"{generate_water_flow_data.__name__};1;{len(data)};{detector_type};{drift_action.__name__};{seq_len};{hidden_sizes};{loss};{mae_full};{mae_warmup}\n"
                        )