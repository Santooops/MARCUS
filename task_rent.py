import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import random

from marcus_new import MarcusModel

# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LR = 5e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 800
PATIENCE = 50
SEED = 42

HIDDEN_DIM = 64
NUM_TIME = 36

REGION_DICT_PATH = "region_dict.pkl"
TRAIN_PATH = "rent_index/train.csv"
VAL_PATH   = "rent_index/val.csv"
TEST_PATH  = "rent_index/test.csv"

# csv 里 y(t+1) 在 y_next
TARGET_COL = "y_next"

# no-rent backbone ckpt
MARCUS_CKPT = "best_marcus_backbone.pt"

OUT_BEST_HEAD = "best_marcus_rentindex.pt"

# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
print(f"[Setup] Seed={SEED}")

# ============================================================
# Dataset 
# ============================================================

class RentIndexDataset(Dataset):
    def __init__(self, region_dict, df, y_col):
        self.region_dict = region_dict
        self.df = df.reset_index(drop=True)
        self.y_col = y_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 输入特征来自 (t) 的 year_month
        key = (int(row["modzcta"]), str(row["year_month"]))
        item = self.region_dict[key]

        return {
            "poi": item["poi"]["missing"],
            "census": item["census"]["missing"],
            "traffic": item["traffic"]["missing"],
            "time_idx": item["time_idx"],
            "y": row[self.y_col],   # y_norm of y_next
        }

def collate_fn(batch):
    return {
        "poi": torch.tensor(np.stack([b["poi"] for b in batch]), dtype=torch.float, device=DEVICE),
        "census": torch.tensor(np.stack([b["census"] for b in batch]), dtype=torch.float, device=DEVICE),
        "traffic": torch.tensor(np.stack([b["traffic"] for b in batch]), dtype=torch.float, device=DEVICE),
        "time_idx": torch.tensor([b["time_idx"] for b in batch], dtype=torch.long, device=DEVICE),
        "y": torch.tensor([b["y"] for b in batch], dtype=torch.float, device=DEVICE),
    }

# ============================================================
# Load Data
# ============================================================

print("[Data] Loading region_dict + csv...")

with open(REGION_DICT_PATH, "rb") as f:
    region_dict = pickle.load(f)

train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)
test_df  = pd.read_csv(TEST_PATH)

# standardize label: fit on train y_next
y_mean = train_df[TARGET_COL].mean()
y_std  = train_df[TARGET_COL].std()

train_df["y_norm"] = (train_df[TARGET_COL] - y_mean) / (y_std + 1e-8)
val_df["y_norm"]   = (val_df[TARGET_COL] - y_mean) / (y_std + 1e-8)
test_df["y_norm"]  = (test_df[TARGET_COL] - y_mean) / (y_std + 1e-8)

print(f"[Data] {TARGET_COL} mean={y_mean:.4f}, std={y_std:.4f}")

train_ds = RentIndexDataset(region_dict, train_df, "y_norm")
val_ds   = RentIndexDataset(region_dict, val_df, "y_norm")
test_ds  = RentIndexDataset(region_dict, test_df, "y_norm")

def worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                          collate_fn=collate_fn, worker_init_fn=worker_init_fn)
val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False,
                          collate_fn=collate_fn, worker_init_fn=worker_init_fn)
test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False,
                          collate_fn=collate_fn, worker_init_fn=worker_init_fn)

print(f"[Data] Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}")

# ============================================================
# Model: backbone (frozen) + regression head
# ============================================================

print("[Model] Loading no-rent Marcus backbone...")

backbone = MarcusModel(hidden_dim=HIDDEN_DIM, num_time=NUM_TIME).to(DEVICE)

ckpt = torch.load(MARCUS_CKPT, map_location=DEVICE)
backbone.load_state_dict(ckpt["model"], strict=True)

backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False

print("[Model] Backbone loaded + frozen")

# Regression head
reg_head = nn.Sequential(
    nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
    nn.BatchNorm1d(HIDDEN_DIM),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
    nn.BatchNorm1d(HIDDEN_DIM // 2),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(HIDDEN_DIM // 2, 1)
).to(DEVICE)

optimizer = Adam(reg_head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=15
)

# ============================================================
# Loss (Huber)
# ============================================================

def huber_loss(pred, target, delta=1.0):
    err = pred - target
    abs_err = torch.abs(err)
    loss = torch.where(
        abs_err < delta,
        0.5 * err ** 2,
        delta * (abs_err - 0.5 * delta)
    )
    return loss.mean()

# ============================================================
# Train
# ============================================================

print(f"\n[Training] LR={LR} WD={WEIGHT_DECAY} Epochs={MAX_EPOCHS}")

best_val = float("inf")
wait = 0

for epoch in range(MAX_EPOCHS):
    reg_head.train()
    total_train = 0.0

    for batch in train_loader:
        with torch.no_grad():
            h = backbone(
                {
                    "poi": batch["poi"],
                    "census": batch["census"],
                    "traffic": batch["traffic"],
                },
                time_idx=batch["time_idx"]
            )

        pred = reg_head(h).squeeze(-1)
        loss = huber_loss(pred, batch["y"], delta=1.0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reg_head.parameters(), 1.0)
        optimizer.step()

        total_train += loss.item()

    train_loss = total_train / len(train_loader)

    # ---- Val ----
    reg_head.eval()
    with torch.no_grad():
        preds, gts = [], []
        for batch in val_loader:
            h = backbone(
                {
                    "poi": batch["poi"],
                    "census": batch["census"],
                    "traffic": batch["traffic"],
                },
                time_idx=batch["time_idx"]
            )
            p = reg_head(h).squeeze(-1)
            preds.append(p)
            gts.append(batch["y"])

        preds = torch.cat(preds)
        gts = torch.cat(gts)

        val_mse = F.mse_loss(preds, gts).item()
        val_mae = F.l1_loss(preds, gts).item()

    scheduler.step(val_mse)

    # ---- Early stopping ----
    if val_mse < best_val - 1e-5:
        best_val = val_mse
        wait = 0
        torch.save(
            {"model": reg_head.state_dict(), "epoch": epoch, "val_mse": val_mse},
            OUT_BEST_HEAD
        )
    else:
        wait += 1

    if epoch % 20 == 0 or wait == 0:
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d} | TrainLoss {train_loss:.4f} | ValMSE {val_mse:.4f} "
              f"| ValMAE {val_mae:.4f} | LR {lr_now:.6f} | Wait {wait}/{PATIENCE}")

    if wait >= PATIENCE:
        print(f"\n[Training] Early stop at epoch {epoch}")
        break

# ============================================================
# Test
# ============================================================

print("\n[Testing] Evaluating best head on test set...")

best = torch.load(
    OUT_BEST_HEAD,
    map_location=DEVICE,
    weights_only=False
)
reg_head.load_state_dict(best["model"])
reg_head.eval()

print(f"[Testing] Loaded best head from epoch {best['epoch']} (val_mse={best['val_mse']:.6f})")

with torch.no_grad():
    preds, gts = [], []
    for batch in test_loader:
        h = backbone(
            {
                "poi": batch["poi"],
                "census": batch["census"],
                "traffic": batch["traffic"],
            },
            time_idx=batch["time_idx"]
        )
        p = reg_head(h).squeeze(-1)
        preds.append(p)
        gts.append(batch["y"])

preds = torch.cat(preds).cpu().numpy()
gts   = torch.cat(gts).cpu().numpy()

# denormalize back to original y_next scale
pred_raw = preds * (y_std + 1e-8) + y_mean
gt_raw   = gts   * (y_std + 1e-8) + y_mean

mae  = mean_absolute_error(gt_raw, pred_raw)
rmse = np.sqrt(mean_squared_error(gt_raw, pred_raw))
r2   = r2_score(gt_raw, pred_raw)

eps = 1e-8
mape = np.mean(np.abs((gt_raw - pred_raw) / (np.abs(gt_raw) + eps))) * 100.0

print("\n" + "="*60)
print("TEST RESULTS (No-Rent Marcus Backbone)  [Predict y(t+1)]")
print("="*60)
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²:   {r2:.4f}")
print("="*60)
print("\n[Done]")