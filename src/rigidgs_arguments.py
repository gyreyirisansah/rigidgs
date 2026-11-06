"""
RigidGS argument groups (cleaned, single-pass trainer with optional Stage 3).

Pipeline this matches:
    Stage 1: Unified photometric + mask BCE + boundary reg + cleanup
    Stage 2: LaMa inpainting at iter `inpaint_at_iter` = `iterations`
    Stage 3: Optional BG refinement on inpainted images (FG frozen)
    Stage 4: Final cleanup + save

Defaults are tuned for a 30000-iteration Stage 1 run on LLFF scenes
(standard 3DGS schedule). For shorter runs, scale `densify_*`,
`opacity_reset_interval`, and `position_lr_max_steps` proportionally.

Removed from prior version (legacy 2-phase trainer dead code):
    cycle_length, n_cycles, start_phase_B_at_iter
    repulsion_weight, surface_weight, repulsion_*
    adoption_regularization_*
    phase_b_densify_*, phase_b_opacity_reset_interval
    resume_full_training_from_iter
    save_all_cycles, keep_all_checkpoints
    inpaint_weight, inpaint_interval, inpaint_until_iter
"""

from argparse import ArgumentParser, Namespace
import os
import sys


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = key.startswith("_")
            if shorthand:
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0]),
                                       default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0]),
                                       default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key,
                                       default=value, action="store_true")
                else:
                    group.add_argument("--" + key,
                                       default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


# =====================================================================
# Scene / dataset loading
# =====================================================================
class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self.scene_name = ""
        self._model_path = ""
        self.images = ""
        self.inpainted_images = ""
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        self.llffhold = 8
        super().__init__(parser, "Loading Parameters", sentinel)


# =====================================================================
# Rasterizer pipeline knobs
# =====================================================================
class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")


# =====================================================================
# Stage 1 optimization — standard 3DGS 30k schedule
# =====================================================================
# These are the canonical 3DGS defaults for high-quality reconstruction.
# 30k iterations gives the photometric loss + densification cycle enough
# time to converge fully. Mask BCE and boundary regularization piggyback
# on this schedule with their own start/until iterations.
# =====================================================================
class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # Total Stage 1 iterations. Stage 3 (refinement) is separate.
        self.iterations = 30_000

        # Position LR (xyz). Standard 3DGS schedule.
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000

        # Other parameter LRs (unchanged from standard 3DGS)
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001

        # Exposure scheduler
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0

        # Photometric loss balance
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2

        # Densification schedule (standard 3DGS 30k)
        self.densification_interval = 100
        self.opacity_reset_interval = 3_000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002

        # Depth loss (unused if depth maps not provided)
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01

        # Misc
        self.random_background = False
        self.optimizer_type = "default"  # or "sparse_adam"

        super().__init__(parser, "Optimization Parameters")


# =====================================================================
# Generic training params — minimal, only what the trainer reads via
# the _get(training_params, name, default) fallback. Most pipeline
# knobs (boundary reg / mask BCE / inpainting / Stage 3) are added
# directly by the trainer's argparse and don't live here.
# =====================================================================
class TrainingParams(ParamGroup):
    def __init__(self, parser):
        # Background color for non-random_background runs
        self.background_color = [0, 0, 0]

        # Logging cadence
        self.log_interval = 100
        self.print_interval = 500
        self.checkpoint_interval = 500

        super().__init__(parser, "Training Parameters")


# =====================================================================
# Resume-from-checkpoint helper (unchanged)
# =====================================================================
def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v
    return Namespace(**merged_dict)



##Margine arguments for the training llff dataset
# Fern 2px 