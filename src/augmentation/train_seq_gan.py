#!/usr/bin/env python3
"""
train_seq_gan.py

Sequence-GAN for (N, T, F) windows.

Example:
  python train_seq_gan.py windows/attack_windows_norm.npz synthetic/attack_synth.npz \
    --scaler models/attack_scaler.pkl --epochs 1500 --batch 4 --gen 800 --save_interval 150 --save_inverse

Notes:
 - Script auto-uses CUDA if available.
 - If a scaler (joblib MinMaxScaler) is provided and --save_inverse is used,
   an inverse-transformed file will be saved alongside the normalized output.
"""
import os
import sys
import argparse
import time
import numpy as np
import joblib
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------- device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- dataset ----------------
class SeqDataset(Dataset):
    def __init__(self, arr):
        self.data = arr.astype("float32")
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return torch.from_numpy(self.data[idx])

# ---------------- models ----------------
class Generator(nn.Module):
    def __init__(self, latent_dim, seq_len, feat_dim, hidden=128, n_layers=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=n_layers, batch_first=True)
        self.out = nn.Linear(hidden, feat_dim)
        self.seq_len = seq_len

    def forward(self, z):
        # z: (B, latent_dim)
        h = torch.relu(self.fc(z))                 # (B, hidden)
        h = h.unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, T, hidden)
        x, _ = self.lstm(h)
        out = torch.sigmoid(self.out(x))  # output in [0,1] to match normalized data
          
        return out

class Discriminator(nn.Module):
    def __init__(self, seq_len, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(feat_dim, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):          # x: (B, T, F)
        return self.net(x.permute(0,2,1))

# ---------------- helpers ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("in_npz", help="input normalized windows npz (data key)")
    p.add_argument("out_npz", help="output file for synthetic npz (normalized)")
    p.add_argument("--scaler", default=None, help="optional scaler pkl for inverse-transform")
    p.add_argument("--save_inverse", action="store_true", help="save inverse-transformed synthetic file when scaler provided")
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--gen", type=int, default=500)
    p.add_argument("--save_interval", type=int, default=200)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()

def print_device_info():
    print("=== Device info ===")
    print("PyTorch version:", torch.__version__)
    print("Torch CUDA runtime:", torch.version.cuda)
    print("Device selected:", device)
    if device.type == "cuda":
        try:
            print("GPU name:", torch.cuda.get_device_name(0))
            print("CUDA device count:", torch.cuda.device_count())
            # memory stats (initial)
            print("GPU reserved (MB):", torch.cuda.memory_reserved(0) / 1024**2)
            print("GPU allocated (MB):", torch.cuda.memory_allocated(0) / 1024**2)
        except Exception:
            pass
    print("===================\n")

def safe_mkdir(d):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ---------------- training ----------------
def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print_device_info()

    # load data
    if not os.path.exists(args.in_npz):
        print("Input file not found:", args.in_npz); sys.exit(1)
    d = np.load(args.in_npz)
    if "data" not in d:
        print("Input npz must contain key 'data' (shape N,T,F). Got keys:", list(d.keys()))
        sys.exit(1)
    X = d["data"]    # shape (N, T, F)
    N, T, F = X.shape
    print(f"Loaded {args.in_npz} -> {X.shape} (N windows, seq_len, feat_dim)")

    dataset = SeqDataset(X)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.num_workers)

    # models
    G = Generator(args.latent, seq_len=T, feat_dim=F).to(device)
    D = Discriminator(seq_len=T, feat_dim=F).to(device)

    optG = optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.9))
    optD = optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.9))
    loss_fn = nn.BCELoss()

    real_label = 1.0; fake_label = 0.0
    safe_mkdir(args.checkpoint_dir)

    try:
        start = time.time()
        for epoch in range(1, args.epochs + 1):
            G.train(); D.train()
            d_losses = []
            g_losses = []

            for real_batch in loader:
                real_batch = real_batch.to(device)        # (B, T, F)
                B = real_batch.size(0)

                # Train Discriminator
                optD.zero_grad()
                labels_real = torch.full((B,1), real_label, device=device)
                out_real = D(real_batch)
                loss_real = loss_fn(out_real, labels_real)

                z = torch.randn(B, args.latent, device=device)
                fake_batch = G(z)
                labels_fake = torch.full((B,1), fake_label, device=device)
                out_fake = D(fake_batch.detach())
                loss_fake = loss_fn(out_fake, labels_fake)

                d_loss = (loss_real + loss_fake) * 0.5
                d_loss.backward()
                optD.step()

                # Train Generator
                optG.zero_grad()
                labels_gen = torch.full((B,1), real_label, device=device)
                out_gen = D(fake_batch)
                # Encourage variance diversity
                variance_loss = -torch.var(fake_batch, dim=0).mean()
                g_loss = loss_fn(out_gen, labels_gen) + 0.5 * variance_loss
                g_loss.backward()
                optG.step()

                d_losses.append(d_loss.item()); g_losses.append(g_loss.item())

            # logging & checkpoint
            if epoch % args.save_interval == 0 or epoch == args.epochs:
                avg_d = float(np.mean(d_losses)) if d_losses else 0.0
                avg_g = float(np.mean(g_losses)) if g_losses else 0.0
                elapsed = time.time() - start
                print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch}/{args.epochs} | D_loss {avg_d:.4f} | G_loss {avg_g:.4f} | elapsed {elapsed/60:.2f} min")
                # GPU mem snapshot
                if device.type == "cuda":
                    try:
                        print("  GPU reserved (MB):", torch.cuda.memory_reserved(0) / 1024**2)
                        print("  GPU allocated (MB):", torch.cuda.memory_allocated(0) / 1024**2)
                        print("  GPU max allocated (MB):", torch.cuda.max_memory_allocated(0) / 1024**2)
                    except Exception:
                        pass
                # save checkpoint
                ckpt = {
                    "epoch": epoch,
                    "G_state": G.state_dict(),
                    "D_state": D.state_dict(),
                    "optG": optG.state_dict(),
                    "optD": optD.state_dict()
                }
                torch.save(ckpt, os.path.join(args.checkpoint_dir, f"gan_ckpt_epoch{epoch}.pth"))
                # free cuda cache
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        total_time = time.time() - start
        print(f"Training finished in {total_time/60.0:.2f} minutes.")

    except RuntimeError as e:
        print("RuntimeError during training (possible OOM).")
        print(str(e))
        traceback.print_exc()
        print("Suggestion: reduce --batch, reduce latent size (--latent), or reduce model hidden sizes.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Training interrupted by user. Will still attempt to generate from current G.")
    except Exception as e:
        print("Unexpected error during training:")
        traceback.print_exc()
        sys.exit(1)

    # generate synthetic
    G.eval()
    synth = []
    with torch.no_grad():
        remaining = args.gen
        batch_gen = min(64, args.gen)
        while remaining > 0:
            b = min(remaining, batch_gen)
            z = torch.randn(b, args.latent, device=device)
            out = G(z).cpu().numpy()
            synth.append(out)
            remaining -= b
    synth = np.vstack(synth)   # (gen, T, F)
    print("Synth shape (normalized):", synth.shape)

    # Save normalized synthetic
    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    np.savez_compressed(args.out_npz, data=synth)
    print("Saved normalized synthetic to:", args.out_npz)

    # optional inverse transform
    if args.save_inverse and args.scaler:
        if os.path.exists(args.scaler):
            try:
                print("Inverse transforming generated sequences using scaler:", args.scaler)
                scaler = joblib.load(args.scaler)
                B,TT,FF = synth.shape
                flat = synth.reshape(-1, FF)
                inv = scaler.inverse_transform(flat)
                synth_inv = inv.reshape(B,TT,FF).astype("float32")
                # save inverse file next to out_npz with suffix _inv.npz
                out_inv = args.out_npz.replace(".npz", "_inv.npz")
                np.savez_compressed(out_inv, data=synth_inv)
                print("Saved inverse-transformed synthetic to:", out_inv)
            except Exception as e:
                print("Failed to inverse-transform synthetic:", str(e))
                traceback.print_exc()
        else:
            print("Scaler file not found, skipping inverse transform:", args.scaler)

def main():
    args = parse_args()
    print("Running train_seq_gan.py")
    print("Command line args:", vars(args))
    train(args)

if __name__ == "__main__":
    main()
