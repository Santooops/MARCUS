import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

from marcus_new import MarcusModel

# ============================================================
# 0) Utils: slice value part (emb only)
# ============================================================

# x = [emb | flag, ratio | loc]
DIMS = {
    "poi":     {"emb": 384, "loc": 13},
    "census":  {"emb": 32,  "loc": 5},   # NYC census LOC_DIM=5
    "traffic": {"emb": 16,  "loc": 2},   # NYC traffic LOC_DIM=2
}

def get_emb_part(x: torch.Tensor, modality: str) -> torch.Tensor:
    """return emb part only: (B, emb_dim)"""
    emb_dim = DIMS[modality]["emb"]
    return x[:, :emb_dim]

def get_miss_dim(modality: str) -> int:
    """2 + loc_dim"""
    return 2 + DIMS[modality]["loc"]


# ============================================================
# 1) Dataset (Temporal sequences)
# ============================================================

class TemporalRegionDataset(Dataset):
    def __init__(self, region_dict, use_version="missing", seq_len=3):
        """
        region_dict key: (region_id, year_month)
        item includes: poi/census/traffic/time_idx
        """
        self.region_dict = region_dict
        self.use_version = use_version
        self.seq_len = seq_len

        self.region_time_map = defaultdict(list)
        for (rid, year_month), item in region_dict.items():
            self.region_time_map[rid].append({
                "time_idx": item["time_idx"],
                "data": item
            })

        for rid in self.region_time_map:
            self.region_time_map[rid].sort(key=lambda x: x["time_idx"])

        self.samples = []
        for rid, ts in self.region_time_map.items():
            if len(ts) >= seq_len:
                for i in range(len(ts) - seq_len + 1):
                    self.samples.append((rid, i))

        print(f"[Dataset] {len(self.samples)} sequences from {len(self.region_time_map)} regions")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rid, start_idx = self.samples[idx]
        ts = self.region_time_map[rid]

        sequence = []
        for i in range(start_idx, start_idx + self.seq_len):
            item = ts[i]["data"]
            sequence.append({
                "poi": item["poi"][self.use_version],
                "census": item["census"][self.use_version],
                "traffic": item["traffic"][self.use_version],
                "time_idx": ts[i]["time_idx"],
            })

        return {"rid": rid, "sequence": sequence}


def collate_fn(batch):
    seq_len = len(batch[0]["sequence"])

    def stack_seq(key):
        # output: (B, T, D)
        stacked = []
        for b in batch:
            stacked.append(np.stack([s[key] for s in b["sequence"]], axis=0))
        return torch.from_numpy(np.stack(stacked, axis=0)).float()

    time_idx = [
        [s["time_idx"] for s in b["sequence"]]
        for b in batch
    ]

    return {
        "poi": stack_seq("poi"),
        "census": stack_seq("census"),
        "traffic": stack_seq("traffic"),
        "time_idx": torch.tensor(time_idx, dtype=torch.long),
        "rid": [b["rid"] for b in batch]
    }


# ============================================================
# 2) Config
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
EPOCHS = 500
LR = 2e-3
PATIENCE = 30
MIN_DELTA = 1e-4

SEQ_LEN = 3

RECON_WEIGHT = 1.0
TEMPORAL_WEIGHT = 0.5
CONTRASTIVE_WEIGHT = 0.3

REGION_DICT_PATH = "NYC_KDD_F/region_dict.pkl"
OUT_CKPT_PATH = "best_marcus_backbone.pt"


# ============================================================
# 3) Load data
# ============================================================

with open(REGION_DICT_PATH, "rb") as f:
    region_dict = pickle.load(f)

dataset = TemporalRegionDataset(region_dict, use_version="missing", seq_len=SEQ_LEN)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    drop_last=True
)

print("Total samples:", len(dataset))


# ============================================================
# 4) Model + Heads
# ============================================================

model = MarcusModel(
    hidden_dim=64,
    num_time=36,
    poi_emb_dim=384, poi_loc_dim=13,
    census_emb_dim=32, census_loc_dim=5,
    traffic_emb_dim=16, traffic_loc_dim=2,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

recon_heads = nn.ModuleDict({
    "census":  nn.Linear(64, DIMS["census"]["emb"]),   # 32
    "traffic": nn.Linear(64, DIMS["traffic"]["emb"]),  # 16
}).to(DEVICE)

# temporal predictor
temporal_predictor = nn.Sequential(
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(128, 64)
).to(DEVICE)

# contrastive projection head
projection_head = nn.Sequential(
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 64)
).to(DEVICE)


# ============================================================
# 5) Loss Functions
# ============================================================

MODALITIES = ["census", "traffic"]

def reconstruction_loss(model, batch_last, time_idx_last):
    """
    batch_last: a dictionary containing {poi, census, traffic}, each with shape (B, D).
    Only reconstruct the embedding part of one specific modality.
    The masked part is also the embedding part, while the missing information features
    (flag, ratio, and location encoding) are preserved.
    """
    modality = np.random.choice(MODALITIES)

    # target: emb part only
    target = get_emb_part(batch_last[modality], modality).clone()

    # masked input: set emb part to 0
    masked_batch = {k: v.clone() for k, v in batch_last.items()}
    emb_dim = DIMS[modality]["emb"]
    masked_batch[modality][:, :emb_dim] = 0.0

    region_emb = model(masked_batch, time_idx=time_idx_last)
    pred = recon_heads[modality](region_emb)

    return F.mse_loss(pred, target)


def temporal_prediction_loss(model, batch_seq, time_idx_seq):
    """
    batch_seq: dict {poi,census,traffic,time_idx} each (B,T,D)
    time_idx_seq: (B,T)
    """
    B, T = time_idx_seq.shape
    if T < 3:
        return torch.tensor(0.0, device=DEVICE)

    embeddings = []
    for t in range(T):
        batch_t = {k: v[:, t] for k, v in batch_seq.items() if k != "time_idx"}
        emb = model(batch_t, time_idx=time_idx_seq[:, t])
        embeddings.append(emb)

    inp = torch.cat([embeddings[-3], embeddings[-2]], dim=-1)
    pred = temporal_predictor(inp)
    target = embeddings[-1]

    return F.mse_loss(pred, target)


def contrastive_loss(model, batch_seq, time_idx_seq, temperature=0.07):
    """
    Use the first and last time steps of the sequence as a positive sample pair (t0, t_last).
    """
    batch_t0 = {k: v[:, 0] for k, v in batch_seq.items() if k != "time_idx"}
    batch_t1 = {k: v[:, -1] for k, v in batch_seq.items() if k != "time_idx"}

    z0 = projection_head(model(batch_t0, time_idx_seq[:, 0]))
    z1 = projection_head(model(batch_t1, time_idx_seq[:, -1]))

    z0 = F.normalize(z0, dim=-1)
    z1 = F.normalize(z1, dim=-1)

    logits = torch.matmul(z0, z1.T) / temperature
    labels = torch.arange(z0.size(0), device=DEVICE)

    return F.cross_entropy(logits, labels)


# ============================================================
# 6) Training Loop
# ============================================================

best_loss = float("inf")
wait = 0

for epoch in range(EPOCHS):
    model.train()
    recon_heads.train()
    temporal_predictor.train()
    projection_head.train()

    total_loss = 0.0

    for batch in loader:
        batch_seq = {k: v.to(DEVICE) for k, v in batch.items() if k != "rid"}

        # last step (B, D)
        batch_last = {k: v[:, -1] for k, v in batch_seq.items() if k != "time_idx"}
        time_idx_last = batch_seq["time_idx"][:, -1]

        loss_recon = reconstruction_loss(model, batch_last, time_idx_last)
        loss_temp = temporal_prediction_loss(model, batch_seq, batch_seq["time_idx"])
        loss_con = contrastive_loss(model, batch_seq, batch_seq["time_idx"])

        loss = (
            RECON_WEIGHT * loss_recon +
            TEMPORAL_WEIGHT * loss_temp +
            CONTRASTIVE_WEIGHT * loss_con
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    improved = (best_loss - avg_loss) > MIN_DELTA
    if improved:
        best_loss = avg_loss
        wait = 0
        torch.save({
            "model": model.state_dict(),
            "recon_heads": recon_heads.state_dict(),
            "temporal_predictor": temporal_predictor.state_dict(),
            "projection_head": projection_head.state_dict(),
            "dims": DIMS,
        }, OUT_CKPT_PATH)
    else:
        wait += 1

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {avg_loss:.6f} | "
            f"Best: {best_loss:.6f} | "
            f"Wait: {wait}/{PATIENCE}"
        )

    if wait >= PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break

print("Training finished.")
print("Saved to:", OUT_CKPT_PATH)