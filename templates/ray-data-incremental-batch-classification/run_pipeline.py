"""Batch entrypoint for job.yaml: the notebook's pipeline, without the comparisons.

Reads today's rows and the keys already classified, splits today's rows into
blocks for the classifier's actors, drops the already-classified rows with the
broadcast probe, classifies the rest with fractional GPUs, and writes
partitioned Parquet with one root _SUCCESS. Generates synthetic input first if
none exists.
"""

import os

import ray

import incremental as inc


def main() -> None:
    # The entrypoint in job.yaml installs the lock on the driver; this hands the same file to
    # every worker. A driver-side install never reaches them.
    lock = os.path.join(os.path.dirname(os.path.abspath(__file__)), "python_depset.lock")
    # A path, not the module object: as a Ray job the driver's runtime_env is deep-copied into
    # the job's, and a module object cannot be copied ("cannot pickle 'module' object").
    ray.init(runtime_env={"pip": lock, "py_modules": [inc.__file__]})
    inc.enable_hang_detection()
    # One stack dump of every driver thread if the job is still running at 50
    # minutes; with job.yaml's 60-minute timeout that leaves a record of where
    # a stalled driver was blocked before the job is killed.
    inc.arm_driver_stack_dump(float(os.getenv("STACK_DUMP_AFTER_S", "3000")))

    storage = os.getenv("STORAGE_DIR", "/mnt/cluster_storage/incremental-demo")
    today_path = os.getenv("TODAY_PATH", f"{storage}/today")
    prior_path = os.getenv("PRIOR_PATH", f"{storage}/prior")
    if not (os.path.isdir(today_path) and os.path.isdir(prior_path)):
        today_df, prior_df = inc.make_synthetic_frames(int(os.getenv("NUM_ROWS", "200000")))
        ray.data.from_pandas(today_df).write_parquet(today_path, mode=ray.data.SaveMode.OVERWRITE)
        ray.data.from_pandas(prior_df).write_parquet(prior_path, mode=ray.data.SaveMode.OVERWRITE)

    gpu_fraction = float(os.getenv("GPU_FRACTION", "0.5"))
    # Split today's rows into blocks for the classifier's actor pool here, before the probe:
    # repartition(num_blocks) waits for its whole input, so after the probe it would hold the
    # GPU stage until the probe finished. The probe keeps one output block per input block, so
    # the classifier receives these blocks, and the probe's own four actors get work too.
    today = inc.split_for_actors(ray.data.read_parquet(today_path), inc.classify_actors(gpu_fraction))
    prior = ray.data.read_parquet(prior_path)
    new_rows = inc.anti_join_probe(today, prior)
    model_dir = inc.download_model_once(inc.MODEL_ID, f"{storage}/models/mdeberta")
    classified = inc.classify(new_rows, model_dir, gpu_fraction=gpu_fraction)
    result = inc.write_partitioned(classified, os.getenv("OUTPUT_PATH", f"{storage}/out"), "shard", marker="root")
    print(result)


if __name__ == "__main__":
    main()
