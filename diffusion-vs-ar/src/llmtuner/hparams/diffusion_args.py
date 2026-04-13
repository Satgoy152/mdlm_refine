from dataclasses import dataclass, field

@dataclass
class DiffusionArguments:
    r"""
    Arguments of Diffusion Models.
    """
    diffusion_steps: int = field(
        default=64,
        metadata={"help": "timesteps of diffusion models."}
    )
    decoding_strategy: str = field(
        default="stochastic0.5-linear",
        metadata={"help": "<topk_mode>-<schedule>"}
    )
    token_reweighting: bool = field(
        default=False,
        metadata={"help": "use focal loss for token-level reweighting"}
    )
    alpha: float = field(
        default=0.25,
        metadata={"help": "for focal loss"}
    )
    gamma: float = field(
        default=2,
        metadata={"help": "for focal loss"}
    )
    time_reweighting: str = field(
        default='original',
        metadata={"help": "use focal loss for token-level reweighting"}
    )
    topk_decoding: bool = field(
        default=False,
        metadata={"help": "use focal loss for token-level reweighting"}
    )
    refine: bool = field(
        default=False,
        metadata={"help": "enable two-pass refine training: model sees its own outputs and learns to correct them"}
    )
    refine_remask_ratio: str = field(
        default="t",
        metadata={"help": "how to determine remask ratio for pass 2. Options: 't' (use masking rate), 'random', or a float like '0.5'"}
    )
    refine_temperature: float = field(
        default=1.0,
        metadata={"help": "Gumbel noise temperature for sampling in refine pass 1. 0 = argmax (no noise)"}
    )
    refine_loss_type: str = field(
        default="sum",
        metadata={"help": "how to combine pass 1 and pass 2 losses. Options: 'sum', 'mean', 'pass2_only'"}
    )
    proseco: bool = field(
        default=False,
        metadata={"help": "when used with refine, pass 2 uses no remasking (ratio=0) and computes loss over the entire sequence"}
    )
    sampling_method: str = field(
        default="vanilla",
        metadata={"help": "sampling method for generation: 'vanilla', 'refine' (unmasking + correction), or 'proseco' (periodic self-correction)"}
    )
    n_correct_per_step: int = field(
        default=50,
        metadata={"help": "max number of token corrections per step in refine sampling"}
    )
    correct_mode: str = field(
        default="topk",
        metadata={"help": "correction mode for refine sampling: 'topk' or 'threshold'"}
    )
    correct_threshold: float = field(
        default=0.01,
        metadata={"help": "minimum log-probability ratio to accept a correction (only used when correct_mode='threshold')"}
    )
    proseco_budget: int = field(
        default=5,
        metadata={"help": "S — number of corrector forward passes per correction round in proseco sampling"}
    )
    proseco_freq: int = field(
        default=5,
        metadata={"help": "omega — run the corrector every omega outer steps in proseco sampling"}
    )

    def __post_init__(self):
        pass
