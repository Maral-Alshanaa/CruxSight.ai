# ============================================================
# CruxSight.ai — Configuration
# ============================================================

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    window_steps:  int   = 12
    horizon_steps: int   = 6
    step_sec:      int   = 10
    min_samples:   int   = 500
    n_features:    int   = 7
    train_ratio:   float = 0.70
    val_ratio:     float = 0.15
    random_seed:   int   = 42
    batch_size:    int   = 64
    num_workers:   int   = 2


@dataclass
class ModelConfig:
    gat_in_feats:  int   = 7
    gat_hidden:    int   = 32     # tuned: Run 4
    gat_heads:     int   = 4
    gat_layers:    int   = 2
    gat_dropout:   float = 0.2
    toc_lambda:    float = 2.0
    toc_gamma:     float = 0.5
    tft_hidden:    int   = 64     # tuned: Run 4
    tft_heads:     int   = 4
    lstm_layers:   int   = 2
    tft_dropout:   float = 0.2
    causal_hidden: int   = 64
    dag_reg:       float = 1.0
    n_patterns:    int   = 8


@dataclass
class TrainingConfig:
    epochs:          int   = 60
    lr:              float = 5e-4   # tuned: Run 4
    weight_decay:    float = 1e-3   # tuned: Run 4
    grad_clip:       float = 1.0
    patience:        int   = 12     # tuned: Run 4
    lambda_pattern:  float = 0.5
    lambda_ttb:      float = 0.3
    lambda_causal:   float = 0.05  # tuned: Run 4
    lambda_sub:      float = 0.05  # tuned: Run 4
    lambda_rcs_sup:  float = 0.3   # RCS supervision term
    fn_weight:       float = 1.5   # tuned: Run 4
    fp_weight:       float = 1.0
    constraint_mult: float = 3.0


@dataclass
class Config:
    data:        DataConfig     = field(default_factory=DataConfig)
    model:       ModelConfig    = field(default_factory=ModelConfig)
    training:    TrainingConfig = field(default_factory=TrainingConfig)
    device:      str            = 'cuda'
    results_dir: str            = ''
    model_dir:   str            = ''

    @classmethod
    def load(cls, path: str) -> 'Config':
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        cfg          = cls()
        cfg.data     = DataConfig(**d['data'])
        cfg.model    = ModelConfig(**d['model'])
        cfg.training = TrainingConfig(**d['training'])
        cfg.device   = d.get('device', 'cuda')
        return cfg
