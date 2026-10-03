import numpy as np, os
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# ------------------- Augmentation Functions -------------------

def jitter(x, sigma=0.02):
    return x + np.random.normal(0, sigma, x.shape)

def scaling(x, sigma=0.05):
    return x * (1.0 + np.random.normal(0, sigma))

def time_warp(x, sigma=0.15, knot=4):
    T,F = x.shape
    tt = np.linspace(0,1,T)
    pts = np.clip(np.sort(np.random.normal(np.linspace(0,1,knot), sigma)), 0, 1)
    f = interp1d(np.linspace(0,1,knot), pts, kind='linear', fill_value="extrapolate")
    new_t = np.clip(f(tt), 0, 1) * (T-1)

    out = np.zeros_like(x)
    for i in range(F):
        interp = interp1d(np.arange(T), x[:,i], kind='linear', fill_value='extrapolate')
        out[:,i] = interp(new_t)
    return out

# ------------------- Folder Structure -------------------

BASE = "."
IN = os.path.join(BASE, "windows")             # normalized windows input
OUT = os.path.join(BASE, "augmented_windows")  # NEW folder OUTSIDE windows

os.makedirs(OUT, exist_ok=True)

ACTIONS = ["attack", "block", "defence"]
mult = 5

# ------------------- Augmentation Loop -------------------

for action in ACTIONS:
    data = np.load(f"{IN}/{action}_windows_norm.npz")["data"]  # (N,T,F)
    aug_list = []

    for seq in data:
        for _ in range(mult):
            s = time_warp(seq, sigma=0.12)
            s = jitter(s, sigma=0.01)
            s = scaling(s, sigma=0.03)
            aug_list.append(s.astype('float32'))

    aug = np.vstack((data, np.stack(aug_list, axis=0)))

    out_path = os.path.join(OUT, f"{action}_windows_aug.npz")
    np.savez_compressed(out_path, data=aug)

    print(f"{action}: saved {aug.shape} to {out_path}")
