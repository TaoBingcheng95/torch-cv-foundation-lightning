import argparse
from pathlib import Path
from typing import Any, Dict

import hydra
import rootutils
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.mnist_module import MNISTLitModule


def _load_model_depends_from_cfg(
    model_cfg_path: Path,
    overrides: list[str] | None = None,
) -> Dict[str, Any]:
    """用 Hydra 从 model config 重建 MNISTLitModule 的所有 ignore 参数。

    与 train.py 走相同的 `hydra.utils.instantiate` 路径，保证实例化语义一致。
    返回的 dict 可直接作为 `MNISTLitModule.load_from_checkpoint(..., **deps)` 的 kwargs。
    """
    cfg_dir = str(model_cfg_path.parent.resolve())
    cfg_name = model_cfg_path.stem  # 如 "mnist"
    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir, job_name="infer"):
        cfg: OmegaConf = compose(config_name=cfg_name, overrides=overrides or [])

    # instantiate 后得到的是已实例化对象：net=SimpleLeNet, optimizer=AdamW, ...
    # 这些对象就是 MNISTLitModule.__init__ 里 ignore 掉的参数，加载时需显式补传。
    return {
        "net": instantiate(cfg.net),
        "optimizer": instantiate(cfg.optimizer),
        "scheduler": instantiate(cfg.scheduler),
        "criterion": instantiate(cfg.criterion),
        "metrics": instantiate(cfg.metrics),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="logs/train/runs/2026-08-21_09-51-49/checkpoints/epoch_008.ckpt",
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--model-cfg",
        type=str,
        default="configs/model/mnist.yaml",
        help="Path to the model config yaml (used to re-instantiate ignored deps).",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # 1) 从 Hydra config 重建被 ignore 的依赖
    deps = _load_model_depends_from_cfg(Path(args.model_cfg))

    # 2) load_from_checkpoint: 用官方推荐方式显式补传 ignore 参数
    #    hyper_parameters 里只保留了 compile(标量)，weights_only=True 友好。
    model = MNISTLitModule.load_from_checkpoint(
        checkpoint_path, map_location=args.device, **deps
    )
    model.eval()
    
    pt_path = checkpoint_path.with_suffix(".pt")
    torch.save({
        "state_dict": model.net.state_dict(),
    }, pt_path)
    # 【主流做法】直接保存模型的状态字典
    # torch.save(model.state_dict(), 'model_weights.pth') 
    # 或者使用更安全的 safetensors 格式（2024年后的新趋势）
    # save_file(model.state_dict(), 'model_weights.safetensors')

    print(f"Loaded model: {type(model).__name__}")
    print(f"net: {type(model.net).__name__}")
    print(f"criterion: {type(model.criterion).__name__}")
