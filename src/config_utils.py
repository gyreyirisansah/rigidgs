"""Shared helpers for turning a configs/<dataset>/<scene>.json scene config
into argparse defaults / CLI flags. Used by src/preprocessing.py,
src/train.py and scripts/run_pipeline.py so all three agree on what a
scene's settings mean, instead of three separate copies of this logic
silently drifting apart.
"""
import json
from pathlib import Path


def load_config(path) -> dict:
    return json.loads(Path(path).read_text())


def as_flags(options: dict, dash: str = "--") -> list:
    """{"mask-grid-size": 4, "mask-edge-restoration": true} -> CLI flags.
    True adds the bare flag, False/None leaves it out."""
    flags = []
    for key, value in options.items():
        if value is False or value is None:
            continue
        flags.append(dash + key)
        if value is not True:
            flags.append(str(value))
    return flags


def apply_config_defaults(parser, config: dict, section: str) -> None:
    """Resolve config[section] (e.g. "preprocessing" or "train") into CLI
    flags and parse them through `parser`, then install the result as the
    parser's new defaults -- so an explicit flag on the real command line
    still overrides it, but an un-given flag now falls back to the scene's
    paper setting instead of the script's generic default.

    This goes through the real argparse machinery (parser.parse_known_args)
    rather than a raw {key.replace('-','_'): value} dict copy, so flags that
    alias to a differently-named dest (e.g. --label-with-sam2-masks ->
    dest label_with_refined_masks) resolve exactly as they would if typed
    on the command line.

    The config's section only carries tunable settings, never required
    path arguments (e.g. train.py's --_images), so this parse must not
    enforce `required` -- that's still enforced by the real parse_args()
    call on the full command line afterwards.
    """
    flags = as_flags(config.get(section, {}), "--")
    required_actions = [a for a in parser._actions if a.required]
    for action in required_actions:
        action.required = False
    try:
        resolved, _ = parser.parse_known_args(flags)
    finally:
        for action in required_actions:
            action.required = True
    parser.set_defaults(**vars(resolved))


def scene_paths(cfg: dict) -> dict:
    out = Path("output") / cfg["dataset"] / cfg["scene"]
    spinnerf = cfg["preprocessing"].get("spinnerf", False)
    refined = not cfg["preprocessing"].get("no-mask-refinement", False)
    return {
        "out": out,
        "model": out / "rigidgs",
        "pcd": out / "point_cloud" / "pcd_data",
        # --spinnerf trains on the undistorted copies preprocessing writes;
        # "train_images" trains on other photos than preprocessing saw (kitchen)
        "images": (out / "images_undistorted" if spinnerf
                   else Path(cfg.get("train_images", cfg["images"]))),
        "masks": out / "preprocessing" / ("refined/refined_masks" if refined
                                          else "segmentation/masks"),
        "split": ["--eval", "--llffhold", "0"] if spinnerf else ["--eval"],
    }
