import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import time
from typing import Tuple, List
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def load_and_split_runs(
    csv_path: str,
    train_runs: int,
    eval_runs: int,
    test_runs: int,
    seed: int,
    cols_to_transform: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = (
        pd.read_csv(csv_path, index_col=0)
          .reset_index(drop=True)
    )
    rng = np.random.default_rng(seed)
    all_runs = df["run"].unique()

    selected_runs = rng.choice(
        all_runs,
        size=train_runs + eval_runs + test_runs,
        replace=False
    )

    train_ids = selected_runs[:train_runs]
    eval_ids  = selected_runs[train_runs:train_runs + eval_runs]
    test_ids  = selected_runs[train_runs + eval_runs:]

    train_data = df[df["run"].isin(train_ids)].copy()
    eval_data  = df[df["run"].isin(eval_ids)].copy()
    test_data  = df[df["run"].isin(test_ids)].copy()

    # --- transforms ---
    log_transform = lambda x: np.log(x + 1e-8)

    for col in cols_to_transform:
        train_data[col] = log_transform(train_data[col])
        eval_data[col]  = log_transform(eval_data[col])
        test_data[col]  = log_transform(test_data[col])

    return train_data, eval_data, test_data

cols_to_transform = ["EIR_true", "phi", "prev_true", "incall"]

train_data, eval_data, test_data = load_and_split_runs(
    csv_path="data/sim_compendia_train/runs/ANC_Simulation_25000_runs_with_under5_phi_new.csv",
    train_runs=15000,
    eval_runs=300,
    test_runs=200,
    seed=42,
    cols_to_transform=cols_to_transform,
)

def build_windows_1d(x, past, future):
    """
    x: (T,) numpy array
    Returns:
      windows: (T, past+future+1)
      mask:    (T, past+future+1)
    """

    T = len(x)
    W = past + future + 1

    # Pad signal
    x_pad = np.pad(x, (past, future), mode="constant")
    mask_pad = np.pad(np.ones(T, dtype=np.float32), (past, future), mode="constant")

    # Build windows by slicing (vectorized)
    windows = np.lib.stride_tricks.sliding_window_view(x_pad, W)
    mask    = np.lib.stride_tricks.sliding_window_view(mask_pad, W)

    return windows.astype(np.float32), mask.astype(np.float32)
def create_multistream_sequences(
    data,
    win_eir=15,
    win_phi=245,
    prev_col="prev_true"
):
    X_eir, M_eir, y_eir = [], [], []
    X_phi, M_phi, y_phi = [], [], []
    y_inc = []

    past_eir = int(0.75 * win_eir)
    future_eir = win_eir - past_eir

    for _, run_data in data.groupby("run"):
        run_data = run_data.reset_index(drop=True)

        # ---- convert once per run ----
        prev = run_data[prev_col].to_numpy(dtype=np.float32)
        eir  = run_data["EIR_true"].to_numpy(dtype=np.float32)
        phi  = run_data["phi"].to_numpy(dtype=np.float32)
        inc  = run_data["incall"].to_numpy(dtype=np.float32)

        # ---- build windows ----
        X_eir_r, M_eir_r = build_windows_1d(prev, past_eir, future_eir)
        X_phi_r, M_phi_r = build_windows_1d(prev, win_phi, 0)

        # ---- inputs ----
        X_eir.append(X_eir_r[..., None])
        M_eir.append(M_eir_r)
        y_eir.append(eir[:, None])

        X_phi.append(X_phi_r[..., None])
        M_phi.append(M_phi_r)
        y_phi.append(phi[:, None])

        # ---- incidence target (aligned with φ timestep) ----
        y_inc.append(inc[:, None])

    # ---- concatenate across runs ----
    X_eir = np.concatenate(X_eir, axis=0)
    M_eir = np.concatenate(M_eir, axis=0)
    y_eir = np.concatenate(y_eir, axis=0)

    X_phi = np.concatenate(X_phi, axis=0)
    M_phi = np.concatenate(M_phi, axis=0)
    y_phi = np.concatenate(y_phi, axis=0)

    y_inc = np.concatenate(y_inc, axis=0)

    # ---- return format ----
    return {
        "eir": (
            torch.from_numpy(X_eir),
            torch.from_numpy(M_eir),
            torch.from_numpy(y_eir)
        ),
        "phi": (
            torch.from_numpy(X_phi),
            torch.from_numpy(M_phi),
            torch.from_numpy(y_phi)
        ),
        "inc": (
            torch.from_numpy(M_eir),   # SAME mask as EIR (aligned with φ)
            torch.from_numpy(y_inc)
        )
    }
class MultiTaskDataset(Dataset):
    def __init__(self, streams):
        self.streams = streams
        self.length = len(streams["eir"][0])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            k: tuple(v[idx] for v in self.streams[k])
            for k in self.streams
        }

class MSConv(nn.Module):
    """
    Multi-branch temporal conv:
      • ultra-low dilation branch  --> micro structure
      • low dilation branch        --> short-term trends
      • high dilation branch       --> long-range structure
    """
    def __init__(
        self,
        in_channels=1,
        channels_per_branch=8
    ):
        super().__init__()

        # -------- ULTRA-LOW dilation branch --------
        self.ultra = nn.Sequential(
            nn.Conv1d(
                in_channels, channels_per_branch,
                kernel_size=3, dilation=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(channels_per_branch),

            nn.Conv1d(
                channels_per_branch, channels_per_branch,
                kernel_size=3, dilation=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(channels_per_branch)
        )

        # -------- LOW dilation branch (medium receptive field) --------
        self.low = nn.Sequential(
            nn.Conv1d(
                in_channels, channels_per_branch,
                kernel_size=5, dilation=2,
                padding=4    # (5-1)/2 * 2
            ),
            nn.ReLU(),
            nn.BatchNorm1d(channels_per_branch),

            nn.Conv1d(
                channels_per_branch, channels_per_branch,
                kernel_size=5, dilation=2,
                padding=4
            ),
            nn.ReLU(),
            nn.BatchNorm1d(channels_per_branch)
        )

        # -------- HIGH dilation branch (long range) --------
        self.high = nn.Sequential(
            nn.Conv1d(
                in_channels, channels_per_branch,
                kernel_size=7, dilation=8,
                padding=24   # SAME
            ),
            nn.ReLU(),
            nn.BatchNorm1d(channels_per_branch),

            nn.Conv1d(
                channels_per_branch, channels_per_branch,
                kernel_size=7, dilation=16,
                padding=48
            ),
            nn.ReLU(),
            nn.BatchNorm1d(channels_per_branch)
        )

    def forward(self, x):
        # x: (B,1,T)
        u = self.ultra(x)
        l = self.low(x)
        h = self.high(x)
        return torch.cat([u, l, h], dim=1)   # (B, 3*channels, T)


class FourierPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512, num_freqs=8):
        super().__init__()
        self.dim = dim
        self.num_freqs = num_freqs

        freqs = 2 ** torch.arange(num_freqs).float()
        self.register_buffer("freqs", freqs)

        self.proj = nn.Linear(2 * num_freqs, dim)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        B, T, _ = x.shape
        device = x.device

        t = torch.linspace(0, 1, T, device=device)  # normalized time
        angles = 2 * np.pi * t[:, None] * self.freqs[None, :]

        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        pe = self.proj(pe)                          # (T, D)
        pe = pe.unsqueeze(0).expand(B, -1, -1)      # (B, T, D)

        return x + pe

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        self.pe = nn.Embedding(max_len, dim)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device)
        pe = self.pe(positions)          # (T, D)
        return pe.unsqueeze(0).expand(B, -1, -1)

class HybridPositionalEncoding(nn.Module):
    def __init__(
        self,
        dim,
        max_len=512,
        num_freqs=8,
        alpha_init=1.0,
        beta_init=1.0
    ):
        super().__init__()

        self.fourier_pe = FourierPositionalEncoding(
            dim=dim,
            max_len=max_len,
            num_freqs=num_freqs
        )

        self.learned_pe = LearnedPositionalEmbedding(
            max_len=max_len,
            dim=dim
        )

        # Learnable scaling factors
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta  = nn.Parameter(torch.tensor(beta_init))

    def forward(self, x):
        """
        x: (B, T, D)
        """
        pe_fourier = self.fourier_pe(x) - x
        pe_learned = self.learned_pe(x)

        return x + self.alpha * pe_fourier + self.beta * pe_learned


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, mask):
        energy = torch.tanh(self.attn(x))
        scores = self.v(energy).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return context

class TemporalEncoder(nn.Module):
    def __init__(
        self,
        input_channels=1,
        conv_channels=8,
        hidden_dim=128,
        num_layers=2,
        use_attention=True,
        positional_encoding: nn.Module | None = None,
    ):
        super().__init__()

        self.use_attention = use_attention
        self.pe = positional_encoding

        self.conv = MSConv(
            in_channels=input_channels,
            channels_per_branch=conv_channels
        )
        conv_dim = conv_channels * 3
        self.input_proj = nn.Linear(conv_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        if use_attention:
            self.attn = SelfAttention(hidden_dim)

    def forward(self, x, mask, return_sequence=False):
        """
        x:    (B, T, 1)
        mask: (B, T)
        """

        # Conv
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.input_proj(x)

        if self.pe is not None:
            x = self.pe(x)

        x = x * mask.unsqueeze(-1)

        x, _ = self.lstm(x)   # (B, T, H)

        if return_sequence:
            return x

        if self.use_attention:
            return self.attn(x, mask)
        else:
            # last *valid* timestep
            idx = mask.sum(dim=1).long() - 1
            return x[torch.arange(len(x)), idx]

class IncidenceHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.attn_eir = SelfAttention(hidden_dim)
        self.attn_phi = SelfAttention(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, h_eir_seq, h_phi_seq, mask_eir, mask_phi):
        z_eir = self.attn_eir(h_eir_seq, mask_eir)
        z_phi = self.attn_phi(h_phi_seq, mask_phi)

        z = torch.cat([z_eir, z_phi], dim=-1)
        return self.mlp(z)


class MultiHeadModel(nn.Module):
    def __init__(self, max_len=256):
        super().__init__()

        #Positional encoding ONLY for EIR
        # eir_pe = HybridPositionalEncoding(
        #     dim=128,
        #     max_len=max_len,
        #     num_freqs=8,
        #     alpha_init=1.0,
        #     beta_init=0.5
        # )

        self.eir = TemporalEncoder(
            hidden_dim=128,
            num_layers=3,
            use_attention=True,
            positional_encoding=None #eir_pe #
        )

        self.phi = TemporalEncoder(
            hidden_dim=128,
            num_layers=3,
            use_attention=True,
            positional_encoding=None
        )

        self.head_eir = nn.Linear(128, 1)
        self.head_phi = nn.Linear(128, 1)

        self.incidence = IncidenceHead(hidden_dim=128)

    def forward(self, batch):
        # ---- pooled latents ----
        h_eir = self.eir(
            batch["eir"][0],
            batch["eir"][1],
            return_sequence=False
        )

        h_phi = self.phi(
            batch["phi"][0],
            batch["phi"][1],
            return_sequence=False
        )

        out_eir = self.head_eir(h_eir)
        out_phi = self.head_phi(h_phi)
        
        # ---- sequence latents ----
        h_eir_seq = self.eir(
            batch["eir"][0],
            batch["eir"][1],
            return_sequence=True
        )
        
        h_phi_seq = self.phi(
            batch["phi"][0],
            batch["phi"][1],
            return_sequence=True
        )
        
        # ---- incidence ----
        out_inc = self.incidence(
            h_eir_seq,
            h_phi_seq,
            batch["eir"][1],
            batch["phi"][1]
        )


        return out_eir, out_phi, out_inc

        
def train_model(
    model,
    train_loader,
    eval_loader=None,
    epochs=20,
    lr=5e-4,
    lambda_inc=0.5
):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        # =======================
        # Train
        # =======================
        model.train()
        eir_l = phi_l = inc_l = 0.0

        for batch in train_loader:
            batch = {
                k: tuple(x.to(device) for x in batch[k])
                for k in batch
            }

            pred_eir, pred_phi, pred_inc = model(batch)

            loss_eir = loss_fn(pred_eir, batch["eir"][2])
            loss_phi = loss_fn(pred_phi, batch["phi"][2])
            loss_inc = loss_fn(pred_inc, batch["inc"][1])

            loss = loss_eir + loss_phi + lambda_inc * loss_inc

            opt.zero_grad()
            loss.backward()
            opt.step()

            eir_l += loss_eir.item()
            phi_l += loss_phi.item()
            inc_l += loss_inc.item()

        n = len(train_loader)

        msg = (
            f"Epoch {epoch+1:02d} | "
            f"Train → "
            f"EIR {eir_l/n:.6f} | "
            f"Phi {phi_l/n:.6f} | "
            f"Inc {inc_l/n:.6f}"
        )

        # Eval (optional)
        if eval_loader is not None:
            model.eval()
            eir_e = phi_e = inc_e = 0.0

            with torch.no_grad():
                for batch in eval_loader:
                    batch = {
                        k: tuple(x.to(device) for x in batch[k])
                        for k in batch
                    }

                    pred_eir, pred_phi, pred_inc = model(batch)

                    eir_e += loss_fn(pred_eir, batch["eir"][2]).item()
                    phi_e += loss_fn(pred_phi, batch["phi"][2]).item()
                    inc_e += loss_fn(pred_inc, batch["inc"][1]).item()

            ne = len(eval_loader)
            torch.save(
            model.state_dict(),
            "src/trained_model/multitask_model_improvedMSConv_HPE_EIR_phi_with_incidence.pth")
            msg += (
                f" | Eval → "
                f"EIR {eir_e/ne:.6f} | "
                f"Phi {phi_e/ne:.6f} | "
                f"Inc {inc_e/ne:.6f}"
            )

        print(msg)

streams_train = create_multistream_sequences(train_data)
streams_eval  = create_multistream_sequences(eval_data)

train_loader = DataLoader(MultiTaskDataset(streams_train), batch_size=32, shuffle=True)
eval_loader  = DataLoader(MultiTaskDataset(streams_eval), batch_size=32)

model = MultiHeadModel()
train_model(model, train_loader, eval_loader)

def infer_single_run(model, run_df, win_eir, win_phi, device):
    """
    Returns predictions and ground truth in ORIGINAL (inverse-log) scale
    """
    streams = create_multistream_sequences(
        run_df,
        win_eir=win_eir,
        win_phi=win_phi
    )

    model.eval()
    with torch.no_grad():
        batch = {
            k: tuple(v.to(device) for v in streams[k])
            for k in streams
        }

        pred_eir, pred_phi, pred_inc = model(batch)

    # ---- inverse log ----
    y_eir_true = inverse_log(streams["eir"][2].numpy())
    y_phi_true = inverse_log(streams["phi"][2].numpy())
    y_inc_true = inverse_log(streams["inc"][1].numpy())

    y_eir_pred = inverse_log(pred_eir.cpu().numpy())
    y_phi_pred = inverse_log(pred_phi.cpu().numpy())
    y_inc_pred = inverse_log(pred_inc.cpu().numpy())

    results = {
        "t": run_df["t"].values[:len(y_eir_pred)] / 365.25,
        "eir": (y_eir_true, y_eir_pred),
        "phi": (y_phi_true, y_phi_pred),
        "inc": (y_inc_true, y_inc_pred),
    }

    return results

def plot_multitask_inference(
    model,
    test_data,
    win_eir=12,
    win_phi=300,
    num_runs=20,
    device="cpu",
    save_dir="plots"
):
    runs = np.random.choice(
        test_data["run"].unique(),
        num_runs,
        replace=False
    )

    ncols = int(np.ceil(np.sqrt(num_runs)))
    nrows = int(np.ceil(num_runs / ncols))

    targets = {
    "eir": ("EIR", "eir"),
    "phi": ("φ", "phi"),
    "inc": ("Incidence", "inc")
}



    for key, (title, _) in targets.items():

        fig, axs = plt.subplots(
            nrows, ncols,
            figsize=(5*ncols, 4*nrows),
            sharey=True
        )
        axs = axs.flatten()
        handles, labels = None, None

        for i, run in enumerate(runs):
            run_df = test_data[test_data["run"] == run].reset_index(drop=True)

            results = infer_single_run(model, run_df, win_eir, win_phi, device )#win_inc,

            t = results["t"]
            y_true, y_pred = results[key]  # inverse-log already applied

            r2  = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)

            axs[i].plot(t, y_true, color="black", label="True")
            axs[i].plot(t, y_pred, color="red", label="Pred")#linestyle="--"

            axs[i].set_title(
                f"Run {run}\nR²={r2:.2f} | MAE={mae:.3f}",
                fontsize=9
            )
            axs[i].grid(True, linestyle="--", alpha=0.6)

            if handles is None:
                handles, labels = axs[i].get_legend_handles_labels()

        # delete unused axes
        for j in range(len(runs), len(axs)):
            fig.delaxes(axs[j])

        fig.suptitle(f"{title} – Test Runs", fontsize=16)
        fig.legend(handles, labels, loc="upper center", ncol=2)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"multitask_{key}_test_runs.png", dpi=300)
        plt.show

inverse_log = lambda y: np.exp(y) - 1e-8

model = MultiHeadModel()
model.load_state_dict(
    torch.load("src/trained_model/multitask_model_improvedMSConv_HPE_EIR_phi_with_incidence.pth", map_location=device)
)
model.to(device)

plot_multitask_inference(
    model,
    test_data,
    win_eir=20,
    win_phi=300,
    num_runs=20,
    device=device
)


def plot_goodness_of_fit(model, test_data, win_eir, win_phi, num_runs=20, device="cpu"):
    np.random.seed(2)  # Set the SAME fixed seed here
    runs = np.random.choice(test_data["run"].unique(), num_runs, replace=False)
    
    # Storage for aggregated points across all selected runs
    all_data = {
        "eir": {"true": [], "pred": []},
        "phi": {"true": [], "pred": []},
        "inc": {"true": [], "pred": []}
    }

    # Collect data using your chained inference logic
    for run in runs:
        run_df = test_data[test_data["run"] == run].reset_index(drop=True)
        res = infer_single_run(model, run_df, win_eir, win_phi, device)
        
        for key in ["eir", "phi", "inc"]:
            all_data[key]["true"].extend(res[key][0])
            all_data[key]["pred"].extend(res[key][1])

    # Plotting setup
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [("eir", "EIR"), ("phi", "$\phi$"), ("inc", "Incall")]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, (key, title) in enumerate(metrics):
        y_true = np.array(all_data[key]["true"])
        y_pred = np.array(all_data[key]["pred"])
        
        # Calculate global R2 for the selected runs
        r2 = r2_score(y_true, y_pred)
        mae=mean_absolute_error(y_true, y_pred)
        
        # Scatter Plot
        axs[i].scatter(y_true, y_pred, alpha=0.3, s=10, color=colors[i], label="Predictions")
        
        # Perfect fit line (Identity line)
        max_val = max(y_true.max(), y_pred.max())
        min_val = min(y_true.min(), y_pred.min())
        axs[i].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Perfect Fit (x=y)")
        
        # Formatting
        axs[i].set_title(f"{title}\nOverall $R^2$: {r2:.3f}, $MAE$:{mae:.3f}", fontsize=14, fontweight='bold')
        axs[i].set_xlabel("Observed (Ground Truth)")
        axs[i].set_ylabel("Predicted")
        axs[i].grid(True, linestyle="--", alpha=0.5)
        axs[i].legend()

    plt.tight_layout()
    plt.savefig("goodness of fit for the select runs with one model.png")
    plt.show()

plot_goodness_of_fit(model, test_data, win_eir=20, win_phi=300, num_runs=200, device=device)


# def collect_train_latents(model, loader):
#     model.eval()
#     latents = []

#     with torch.no_grad():
#         for batch in loader:
#             batch = {
#                 k: tuple(x.to(device) for x in batch[k])
#                 for k in batch
#             }

#             h_eir = model.eir(batch["eir"][0], batch["eir"][1], return_sequence=False)
#             h_phi = model.phi(batch["phi"][0], batch["phi"][1], return_sequence=False)

#             z = torch.cat([h_eir, h_phi], dim=-1)  # (B, 256)
#             latents.append(z.cpu())

#     return torch.cat(latents, dim=0)

# def fit_latent_gaussian(latents):

#     mu = latents.mean(dim=0)
#     centered = latents - mu

#     cov = centered.T @ centered / (len(latents) - 1)
#     cov += 1e-5 * torch.eye(cov.shape[0])  # regularization

#     cov_inv = torch.inverse(cov)

#     return mu, cov_inv

# train_latents = collect_train_latents(model, train_loader)
# mu_latent, cov_inv_latent = fit_latent_gaussian(train_latents)

# train_distances = torch.sum(
#     (train_latents - mu_latent) @ cov_inv_latent *
#     (train_latents - mu_latent),
#     dim=1
# )

# ood_threshold = torch.quantile(train_distances, 0.99)

# print("OOD threshold:", ood_threshold.item())

# stats = torch.load("latent_stats.pt")
# mu_latent = stats["mu_latent"]
# cov_inv_latent = stats["cov_inv_latent"]
# ood_threshold = stats["ood_threshold"]


# def overlay_ood(ax, t, mask):
#     if mask is None:
#         return

#     start = None

#     for i in range(len(mask)):
#         if mask[i] and start is None:
#             start = t[i]

#         elif not mask[i] and start is not None:
#             ax.axvspan(start, t[i], alpha=0.2)
#             start = None

#     if start is not None:
#         ax.axvspan(start, t[-1], alpha=0.2)

# def infer_single_run(
#     model,
#     run_df,
#     win_eir,
#     win_phi,
#     device,
#     mu_latent=None,
#     cov_inv_latent=None,
#     ood_threshold=None
# ):
#     """
#     Returns:
#         predictions in ORIGINAL scale
#         + timewise Mahalanobis OOD distance + mask
#     """

#     streams = create_multistream_sequences(
#         run_df,
#         win_eir=win_eir,
#         win_phi=win_phi
#     )

#     model.eval()

#     with torch.no_grad():

#         batch = {
#             k: tuple(v.to(device) for v in streams[k])
#             for k in streams
#         }

#         # ---- forward pass ----
#         pred_eir, pred_phi, pred_inc = model(batch)

        
#         if mu_latent is not None and cov_inv_latent is not None:
        
#             # pooled representations (already aligned correctly)
#             h_eir = model.eir(
#                 batch["eir"][0],
#                 batch["eir"][1],
#                 return_sequence=False
#             )
        
#             h_phi = model.phi(
#                 batch["phi"][0],
#                 batch["phi"][1],
#                 return_sequence=False
#             )
        
#             z = torch.cat([h_eir, h_phi], dim=-1)  # (T, D)
        
#             z_cpu = z.cpu()
#             diff = z_cpu - mu_latent
        
#             d = torch.sum(diff @ cov_inv_latent * diff, dim=1)
        
#             ood_distance = d.numpy()
        
#             if ood_threshold is not None:
#                 ood_mask = ood_distance > ood_threshold.item()
#             else:
#                 ood_mask = None

#     # ---- inverse transforms ----
#     y_eir_true = inverse_log(streams["eir"][2].numpy())
#     y_phi_true = inverse_log(streams["phi"][2].numpy())
#     y_inc_true = inverse_log(streams["inc"][1].numpy())

#     y_eir_pred = inverse_log(pred_eir.cpu().numpy())
#     y_phi_pred = inverse_log(pred_phi.cpu().numpy())
#     y_inc_pred = inverse_log(pred_inc.cpu().numpy())

#     results = {
#         "t": run_df["t"].values[:len(y_eir_pred)] / 365.25,
#         "eir": (y_eir_true, y_eir_pred),
#         "phi": (y_phi_true, y_phi_pred),
#         "inc": (y_inc_true, y_inc_pred),
#         "ood_distance": ood_distance,
#         "ood_mask": ood_mask
#     }

#     return results

# def plot_multitask_inference(
#     model,
#     test_data,
#     mu_latent,
#     cov_inv_latent,
#     ood_threshold,
#     win_eir=20,
#     win_phi=60,
#     num_runs=20,
#     device="cpu",
#     save_dir="plots",
#     show_ood=True
# ):

#     runs = np.random.choice(
#         test_data["run"].unique(),
#         num_runs,
#         replace=False
#     )

#     ncols = int(np.ceil(np.sqrt(num_runs)))
#     nrows = int(np.ceil(num_runs / ncols))

#     targets = {
#         "eir": ("EIR", "eir"),
#         "phi": ("φ", "phi"),
#         "inc": ("Incidence", "inc")
#     }

#     for key, (title, _) in targets.items():

#         fig, axs = plt.subplots(
#             nrows, ncols,
#             figsize=(5*ncols, 4*nrows),
#             sharey=True
#         )
#         axs = axs.flatten()
#         handles, labels = None, None

#         for i, run in enumerate(runs):

#             run_df = test_data[test_data["run"] == run].reset_index(drop=True)

#             results = infer_single_run(
#                 model,
#                 run_df,
#                 win_eir,
#                 win_phi,
#                 device,
#                 mu_latent,
#                 cov_inv_latent,
#                 ood_threshold
#             )

#             t = results["t"]
#             y_true, y_pred = results[key]

#             r2  = r2_score(y_true, y_pred)
#             mae = mean_absolute_error(y_true, y_pred)

#             ood_distance = results["ood_distance"]
#             ood_mask = results["ood_mask"]

#             # ========================
#             # OOD score (run-level)
#             # ========================
#             if ood_distance is not None:
#                 ood_score = np.mean(ood_distance)
#                 is_ood = ood_score > ood_threshold.item()
#             else:
#                 ood_score = np.nan
#                 is_ood = False

#             # ========================
#             # Plot
#             # ========================
#             axs[i].plot(t, y_true, color="black", label="True")
#             axs[i].plot(t, y_pred, color="red", label="Pred")

#             # 🔥 overlay timewise OOD
#             if show_ood:
#                 overlay_ood(axs[i], t, ood_mask)

#             title_color = "red" if is_ood else "black"

#             axs[i].set_title(
#                 f"Run {run}\n"
#                 f"R²={r2:.2f} | MAE={mae:.3f}\n"
#                 f"OOD={ood_score:.2f}",
#                 fontsize=9,
#                 color=title_color
#             )

#             axs[i].grid(True, linestyle="--", alpha=0.6)

#             if handles is None:
#                 handles, labels = axs[i].get_legend_handles_labels()

#         # remove unused axes
#         for j in range(len(runs), len(axs)):
#             fig.delaxes(axs[j])

#         fig.suptitle(f"{title} – Test Runs", fontsize=16)
#         fig.legend(handles, labels, loc="upper center", ncol=2)

#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#         plt.savefig(f"{save_dir}/multitask_{key}_test_runs.png", dpi=300)
#         plt.show()

# plot_multitask_inference(
#     model,
#     test_data,
#     mu_latent,
#     cov_inv_latent,
#     ood_threshold,
#     win_eir=20,
#     win_phi=300,
#     num_runs=6,
#     device=device,
#     show_ood=True
# )