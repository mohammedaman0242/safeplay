# normalize_windows.py
import os, numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

BASE="."
IN = os.path.join(BASE,"sample_videos/windows")
OUT = os.path.join(BASE,"sample_videos/windows")
ACTIONS=["attack","block","defence"]

for action in ACTIONS:
    p=os.path.join(IN,f"{action}_windows.npz")
    d=np.load(p)
    X=d["data"]  # (N, T, F)
    N,T,F = X.shape
    flat = X.reshape(-1, F)
    scaler = MinMaxScaler().fit(flat)
    flat_scaled = scaler.transform(flat).astype('float32')
    Xs = flat_scaled.reshape(N,T,F)
    np.savez_compressed(os.path.join(OUT,f"{action}_windows_norm.npz"), data=Xs, seq_len=T, feat_dim=F)
    joblib.dump(scaler, os.path.join(OUT,f"{action}_scaler.pkl"))
    print("Normalized and saved:", action, Xs.shape)
