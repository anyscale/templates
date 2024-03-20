import os

import ray.train
from ray.train.torch import TorchTrainer

from train_dreambooth_lora_sdxl import main, parse_args


def train_fn_per_worker(config: dict):
    args = config["args"]

    # See `train_dreambooth_lora_sdxl` for all of the actual training details.
    main(args)

    def get_latest_checkpoint_dir():
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        return os.path.join(args.output_dir, path)

    checkpoint_dir = get_latest_checkpoint_dir()
    if checkpoint_dir is not None:
        ray.train.report(
            {},
            checkpoint=ray.train.Checkpoint.from_directory(checkpoint_dir),
        )


def launch_finetuning(input_args=None) -> ray.train.Result:
    args = parse_args(input_args=input_args)

    trainer = TorchTrainer(
        train_fn_per_worker,
        # Pass arguments from the driver to the `config` dict of the `train_fn_per_worker`
        train_loop_config={"args": args},
        scaling_config=ray.train.ScalingConfig(
            num_workers=1,
            use_gpu=True,
            accelerator_type="A10G",
        ),
        run_config=ray.train.RunConfig(
            storage_path=os.environ["ANYSCALE_ARTIFACT_STORAGE"] + "/training_results",
            name="stable-diffusion-finetuning",
        )
    )
    result = trainer.fit()
    return result


if __name__ == "__main__":
    launch_finetuning()

