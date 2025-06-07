import numpy as np
from tqdm import tqdm

from datastreams import (generate_ar1_drift, generate_seasonal_drift,
                         generate_trend_drift)
from models import *


def make_dataset(data, seq_len):
    time_series = []
    for i in range(len(data) - seq_len + 1):
        time_series.append(data[i : i + seq_len])
    return np.array(time_series)[:-1], data[seq_len:]


def train_model(model, x_all, y_all, seq_len, optimizer, loss_fn, device):

    model.train()
    all_loss = []

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

        mae_full = torch.abs(pred - y_ten).mean().item()
        mae_values_full.append(mae_full)

        # Post-warmup MAE tracking
        if step >= 200:
            mae_values.append(mae_full)

    avg_loss = sum(all_loss) / len(all_loss) if all_loss else 0.0
    avg_mae_post_warmup = sum(mae_values) / len(mae_values) if mae_values else 0.0
    avg_mae_full = (
        sum(mae_values_full) / len(mae_values_full) if mae_values_full else 0.0
    )

    return model, avg_loss, avg_mae_post_warmup, avg_mae_full


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    streams = [generate_trend_drift, generate_seasonal_drift, generate_ar1_drift]
    n_dims = [1, 3, 5]
    n_pointss = [20_000, 50_000]
    seq_lens = [16, 32, 64]
    d_models = [16, 32, 64]
    for stream_func in streams:
        for n_dim in tqdm(n_dims):
            for n_points in n_pointss:
                data = stream_func(n_periods=n_points, n_dims=n_dim)
                for seq_len in seq_lens:
                    for d_model in d_models:
                        x_all, y_all = make_dataset(data, seq_len)
                        model = TimeSeriesTransformer(
                            input_size=n_dim,
                            d_model=d_model,
                            nhead=2,
                            num_layers=2,
                            output_size=n_dim,
                        )
                        model.to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                        loss_fn = nn.MSELoss()
                        model, loss, mae_full, mae_warmup = train_model(
                            model,
                            x_all,
                            y_all,
                            seq_len,
                            optimizer,
                            loss_fn,
                            device,
                        )
                        # save stream, n_dim, n_points, detector_type, seq_len, hidden_sizes, train_loss, test_loss to CVS file
                        with open("../results/results_transformer.csv", "a") as f:
                            f.write(
                                f"{stream_func.__name__};{n_dim};{n_points};None;Baseline;{seq_len};{d_model};{loss};{mae_full};{mae_warmup}\n"
                            )
