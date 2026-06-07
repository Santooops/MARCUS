import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Utils: split x = [emb | flag, ratio, loc]
# ============================================================

def split_emb_and_missing(x: torch.Tensor, emb_dim: int, loc_dim: int):
    """
    x: (B, emb_dim + 2 + loc_dim)
    returns:
      x_emb:     (B, emb_dim)
      miss_fr:   (B, 2)              # [flag, ratio]
      miss_loc:  (B, loc_dim)         # multi-hot
      miss_all:  (B, 2 + loc_dim)     # [flag, ratio, loc...]
    """
    x_emb = x[:, :emb_dim]
    miss_fr = x[:, emb_dim: emb_dim + 2]
    miss_loc = x[:, emb_dim + 2: emb_dim + 2 + loc_dim]
    miss_all = x[:, emb_dim: emb_dim + 2 + loc_dim]
    return x_emb, miss_fr, miss_loc, miss_all


# ============================================================
# 1) Intra Block: [emb] + [flag,ratio,loc] -> hidden
# ============================================================

class EmbedIntraBlock(nn.Module):
    def __init__(self, emb_dim: int, loc_dim: int, hidden_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.loc_dim = loc_dim
        self.miss_dim = 2 + loc_dim

        self.value_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
        )
        self.missing_proj = nn.Sequential(
            nn.Linear(self.miss_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        x_emb, _, _, miss_all = split_emb_and_missing(x_raw, self.emb_dim, self.loc_dim)
        h_val = self.value_proj(x_emb)
        h_miss = self.missing_proj(miss_all)
        return self.fuse(torch.cat([h_val, h_miss], dim=-1))


# ============================================================
# 2) Intra Encoder (POI + Census + Traffic)
# ============================================================

class IntraEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        poi_emb_dim: int = 384, poi_loc_dim: int = 13,
        census_emb_dim: int = 32, census_loc_dim: int = 5,
        traffic_emb_dim: int = 16, traffic_loc_dim: int = 2,
    ):
        super().__init__()
        self.poi_block = EmbedIntraBlock(poi_emb_dim, poi_loc_dim, hidden_dim)
        self.census_block = EmbedIntraBlock(census_emb_dim, census_loc_dim, hidden_dim)
        self.traffic_block = EmbedIntraBlock(traffic_emb_dim, traffic_loc_dim, hidden_dim)

    def forward(self, batch_dict: dict) -> dict:
        return {
            "poi": self.poi_block(batch_dict["poi"]),
            "census": self.census_block(batch_dict["census"]),
            "traffic": self.traffic_block(batch_dict["traffic"]),
        }


# ============================================================
# 3) Inter-modality Encoder
#    reliability: use only (flag, ratio) -> scalar in [0,1]
# ============================================================

class InterAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, reliability):
        """
        x: (B, M, H)
        reliability: (B, M)
        """
        attn_out, _ = self.attn(x, x, x)
        attn_out = attn_out * reliability.unsqueeze(-1)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class InterEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.modalities = ["poi", "census", "traffic"]
        self.layers = nn.ModuleList([InterAttentionBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.reliability_proj = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())  # (flag, ratio) -> scalar

    def forward(self, h_dict, missing_fr_dict):
        """
        h_dict: dict of (B, H)
        missing_fr_dict: dict of (B, 2)   # [flag, ratio]
        """
        x = torch.stack([h_dict[m] for m in self.modalities], dim=1)  # (B,M,H)
        reliability = torch.cat([self.reliability_proj(missing_fr_dict[m]) for m in self.modalities], dim=1)  # (B,M)

        for layer in self.layers:
            x = layer(x, reliability)

        return {m: x[:, i, :] for i, m in enumerate(self.modalities)}


# ============================================================
# 4) Missing-aware + Time-aware Fusion
#    gate input: [flag,ratio,loc] + time_context
# ============================================================

class MissingTimeFusion(nn.Module):
    def __init__(self, hidden_dim: int, miss_dims: dict, modalities=("poi", "census", "traffic")):
        super().__init__()
        self.modalities = modalities
        self.gates = nn.ModuleDict()

        for m in modalities:
            in_dim = miss_dims[m] + hidden_dim
            self.gates[m] = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

    def forward(self, h_dict, missing_all_dict, time_context):
        weighted, weights = [], []

        for m in self.modalities:
            gate_in = torch.cat([missing_all_dict[m], time_context], dim=-1)
            w = self.gates[m](gate_in)  # (B,1)
            weighted.append(h_dict[m] * w)
            weights.append(w)

        weighted = torch.stack(weighted, dim=0).sum(dim=0)  # (B,H)
        weights = torch.stack(weights, dim=0).sum(dim=0)    # (B,1)
        return weighted / (weights + 1e-6)


# ============================================================
# 5) Final Marcus Model (NYC aligned)
# ============================================================

class MarcusModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        inter_layers: int = 2,
        inter_heads: int = 4,
        use_inter: bool = True,
        use_fusion: bool = True,

        # NYC time range: 2022-01 ~ 2024-12 => 36 months
        num_time: int = 36,
        time_dim: int = 8,

        # NYC dims (from your export log)
        poi_emb_dim: int = 384, poi_loc_dim: int = 13,
        census_emb_dim: int = 32, census_loc_dim: int = 5,
        traffic_emb_dim: int = 16, traffic_loc_dim: int = 2,
    ):
        super().__init__()
        self.use_inter = use_inter
        self.use_fusion = use_fusion

        self.dims = {
            "poi": (poi_emb_dim, poi_loc_dim),
            "census": (census_emb_dim, census_loc_dim),
            "traffic": (traffic_emb_dim, traffic_loc_dim),
        }
        self.miss_dims = {m: 2 + loc for m, (_, loc) in self.dims.items()}

        self.intra = IntraEncoder(
            hidden_dim=hidden_dim,
            poi_emb_dim=poi_emb_dim, poi_loc_dim=poi_loc_dim,
            census_emb_dim=census_emb_dim, census_loc_dim=census_loc_dim,
            traffic_emb_dim=traffic_emb_dim, traffic_loc_dim=traffic_loc_dim,
        )

        if use_inter:
            self.inter = InterEncoder(hidden_dim=hidden_dim, num_heads=inter_heads, num_layers=inter_layers)

        if use_fusion:
            self.fusion = MissingTimeFusion(hidden_dim, self.miss_dims, modalities=("poi", "census", "traffic"))

        self.time_emb = nn.Embedding(num_time, time_dim)
        self.time_proj = nn.Sequential(nn.Linear(time_dim, hidden_dim), nn.ReLU())

    def forward(self, batch_dict, time_idx=None, return_h=False):
        """
        batch_dict keys: "poi", "census", "traffic"
        each tensor: (B, emb_dim + 2 + loc_dim)
        """
        # ---- Intra ----
        h_dict = self.intra(batch_dict)

        # ---- Missing extraction ----
        missing_fr = {}
        missing_all = {}
        for m in ["poi", "census", "traffic"]:
            emb_dim, loc_dim = self.dims[m]
            _, fr, _, allm = split_emb_and_missing(batch_dict[m], emb_dim, loc_dim)
            missing_fr[m] = fr
            missing_all[m] = allm

        # ---- Inter ----
        if self.use_inter:
            h_dict = self.inter(h_dict, missing_fr)

        # ---- Time context ----
        B = next(iter(h_dict.values())).shape[0]
        H = next(iter(h_dict.values())).shape[1]
        device = next(iter(h_dict.values())).device

        if time_idx is not None:
            t_ctx = self.time_proj(self.time_emb(time_idx))  # (B,H)
        else:
            t_ctx = torch.zeros((B, H), device=device)

        # ---- Fusion ----
        if self.use_fusion:
            region_emb = self.fusion(h_dict, missing_all, t_ctx)
        else:
            region_emb = torch.stack([h_dict[m] for m in ["poi", "census", "traffic"]], dim=0).mean(dim=0)

        if return_h:
            return region_emb, h_dict
        return region_emb