"""
Fine-tune OpenFold3 on OpenBind EV-A71 2A fragment structures with Ray Train.

    python scripts/prepare_finetune.py                              # once: build the training set
    python finetune_ray_train.py                                    # 2 L4 GPUs, 150 steps, about 30 min
    python finetune_ray_train.py --num-workers 4 --max-steps 300 --lr 2e-4

Starts from the OpenBind-0 weights and trains on the 66 fragment-screen complexes
written by scripts/prepare_finetune.py. OF3's own training stack does the work (data module,
weighted sampler, losses, per-sample gradient clipping, EMA); Ray Train adds:

    TorchTrainer      one worker per GPU, NCCL process group, placement on the cluster
    RayDDPStrategy    Lightning DDP driven by Ray instead of torchrun or SLURM
    ray.train.report  loss terms, protein-ligand lDDT, step time and GPU memory every step
    checkpoint        final weights reported to Ray and written under --storage

The final weights are a plain OF3 state dict, the same format as the OpenBind-0
checkpoint, so `run_openfold predict --inference-ckpt-path <file>` loads them directly.
"""

import argparse
import json
import math
import os
import sys
import threading
import tempfile
import time
from pathlib import Path

import yaml

import ray
import ray.train
from ray.train import CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from openfold3.entry_points.validator import TrainingExperimentConfig

SHARED = Path(os.environ.get("AIDDD_SHARED", "/mnt/user_storage/aiddd"))
CONFIG = Path(__file__).parent / "configs" / "finetune_ev71.yml"
DATA = SHARED / "finetune" / "ev71-2a"
DATASET = "openbind-ev71-2a"
# Starting weights. OF3-p2 checkpoints only load in OpenFold3 0.4.x, so a p2 run trains
# in the legacy install: the driver re-executes itself there, and the Ray Train workers
# are started with the same interpreter.
LEGACY_PY = "/opt/of3-legacy/bin/python"
BASES = {
    "ob0": (SHARED / "of3_cache" / "of3-ob-2025-06-30-174k.pt", None),   # OpenBind-0
    "p2": (SHARED / "of3_cache" / "of3-p2-145k.pt", LEGACY_PY),          # start of the OpenBind paper's fine-tune
}
# Short tags for the run name, by training set directory (scripts/prepare_finetune.py)
DATA_TAGS = {"ev71-2a": "evfrag", "ev71-2a-frag80": "evfrag80", "ev71-2a-frag-early": "evfragearly", "ev71-2a+sars2-mpro-fragment": "evsarsfrag",
             "ev71-2a+sars2-mpro-fragment+early": "evsarsearly"}


def dataset_paths(out: Path, cache: str = "training_cache.json") -> dict:
    """The `dataset_paths` block of the OF3 training config for this set."""
    return {"dataset_cache_file": str(out / cache),
            "alignments_directory": str(out / "alignments"),
            "target_structures_directory": str(out / "structures"),
            "target_structure_file_format": "npz",
            "reference_molecule_directory": str(out / "reference_mols")}


def legacy_cache(data: Path) -> str:
    """A copy of the training cache that OpenFold3 0.4.x can read.

    scripts/prepare_finetune.py runs on OpenFold3 0.5, which adds `fallback_conformer_source`
    to every reference molecule. 0.4.x rejects the unknown field; it only has the RDKit
    fallback, which is what every entry asks for, so dropping the field changes nothing."""
    name = "training_cache_of3v04.json"
    src, dst = data / "training_cache.json", data / name
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        cache = json.loads(src.read_text())
        for mol in cache["reference_molecule_data"].values():
            if mol.pop("fallback_conformer_source", "rdkit") != "rdkit":
                raise SystemExit(f"{src}: a non-RDKit conformer fallback cannot be trained in OpenFold3 0.4.x")
        tmp = dst.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(cache, indent=1))   # the 0.4.x reader finds "_type" on line 2
        tmp.rename(dst)
    return name


def run_name(args) -> str:
    """ft-<base>-<data>-<steps>s-<gpus>g[-<n>samp][-<gpu>]-<MMDD-HHMM>, e.g. ft-p2-evfrag-1000s-8g-1023-0915.
    The diffusion samples and GPU type appear only when they differ from the defaults."""
    data = DATA_TAGS.get(args.data.name, args.data.name)
    extra = (f"-{args.diffusion_samples}samp" if args.diffusion_samples != 8 else "") + \
            (f"-{args.accelerator}" if args.accelerator != "L4" else "")
    return (f"ft-{args.base}-{data}-{args.max_steps}s-{args.num_workers}g{extra}-"
            f"{time.strftime('%m%d-%H%M')}")
# Loss terms averaged over workers and reported every step (names as logged by OF3)
REPORTED = ["loss", "diffusion_loss", "distogram_loss", "plddt_loss", "pae_loss",
            "lddt_inter_protein_ligand_step"]


def build_config(args) -> dict:
    """The runner YAML plus everything that depends on paths and flags."""
    cfg = yaml.safe_load(CONFIG.read_text())
    args.name = args.name or run_name(args)
    cfg["experiment_settings"]["restart_checkpoint_path"] = str(base_ckpt(args))
    if args.base == "p2":
        # of3-p2-145k predates the model's `version_tensor` buffer, the one key it lacks
        # (every weight matches), so strict loading would refuse it
        cfg["experiment_settings"]["ckpt_load_settings"]["strict_loading"] = False
    cfg["experiment_settings"]["output_dir"] = str(args.storage / args.name / "of3")
    cfg["pl_trainer_args"].update(devices=1, num_nodes=args.num_workers)
    settings = cfg["model_update"]["custom"]["settings"]
    settings["optimizer"]["learning_rate"] = args.lr

    # OF3's scheduler warms up, plateaus, then decays stepwise and exponentially:
    # lr = peak * decay_factor ** (steps_since_decay // decay_every + 1). Its defaults
    # only start decaying after 100k steps, so a fine-tune of a few hundred steps would
    # end at the peak learning rate -- the worst possible moment to save the weights.
    # Scale the schedule to this run instead: plateau for the first fifth, then decay
    # to about --lr-final-frac of the peak by the last step.
    sched = settings["lr_scheduler"]
    sched["warmup_no_steps"] = args.warmup_steps
    start = max(args.warmup_steps, args.max_steps // 5)
    sched["start_decay_after_n_steps"] = start
    factor = sched.setdefault("decay_factor", 0.95)
    n_decays = max(1, round(math.log(args.lr_final_frac) / math.log(factor)))
    sched["decay_every_n_steps"] = max(1, (args.max_steps - start) // n_decays)

    cfg["model_update"]["custom"]["architecture"]["shared"]["diffusion"]["no_samples"] = args.diffusion_samples
    cfg["data_module_args"]["num_workers"] = args.loader_workers
    cfg["data_module_args"]["batch_size"] = args.batch_size
    cache = legacy_cache(args.data) if args.base == "p2" else "training_cache.json"
    cfg["dataset_paths"][DATASET] = dataset_paths(args.data, cache)
    return cfg


def ray_lightning():
    """Ray's Lightning strategy, environment and prepare_trainer for `pytorch_lightning`.

    OF3 is built on `pytorch_lightning`. Ray's helpers subclass `lightning.pytorch` when
    that package is also installed, and the two are not interchangeable, so in that case
    this falls back to equivalent subclasses of the `pytorch_lightning` classes."""
    import pytorch_lightning as pl
    from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment, prepare_trainer

    if issubclass(RayDDPStrategy, pl.strategies.DDPStrategy):
        return RayDDPStrategy, RayLightningEnvironment, prepare_trainer

    from pytorch_lightning.plugins.environments import LightningEnvironment

    class Strategy(pl.strategies.DDPStrategy):
        @property
        def root_device(self):
            return ray.train.torch.get_device()

        @property
        def distributed_sampler_kwargs(self):
            return dict(num_replicas=self.world_size, rank=self.global_rank)

    class Environment(LightningEnvironment):
        def world_size(self):
            return ray.train.get_context().get_world_size()

        def global_rank(self):
            return ray.train.get_context().get_world_rank()

        def local_rank(self):
            return ray.train.get_context().get_local_rank()

        def node_rank(self):
            return ray.train.get_context().get_node_rank()

        def set_world_size(self, size):
            pass

        def set_global_rank(self, rank):
            pass

        def teardown(self):
            pass

    return Strategy, Environment, lambda trainer: trainer


def train_loop_per_worker(config: dict):
    """Runs on every GPU worker: OF3's training runner with Ray's Lightning plumbing."""
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.loggers import CSVLogger

    from openfold3.core.utils.callbacks import RankSpecificSeedCallback
    from openfold3.entry_points.experiment_runner import TrainingExperimentRunner

    # OpenFold3 0.4.x builds its scheduler without decay_every_n_steps, so the default
    # (50000) applies and the LR stops decaying after its first step. Make the configured
    # value the default; 0.5 passes it explicitly, so this changes nothing there.
    import openfold3.projects.of3_all_atom.runner as of3_runner
    every = config["of3"]["model_update"]["custom"]["settings"]["lr_scheduler"]["decay_every_n_steps"]

    class LRScheduler(of3_runner.AlphaFoldLRScheduler):
        def __init__(self, *a, decay_every_n_steps=every, **kw):
            super().__init__(*a, decay_every_n_steps=decay_every_n_steps, **kw)

    of3_runner.AlphaFoldLRScheduler = LRScheduler

    RayDDPStrategy, RayLightningEnvironment, prepare_trainer = ray_lightning()
    ctx = ray.train.get_context()
    rank = ctx.get_world_rank()

    # After a worker-group restart (a spot node reclaimed, say) Ray hands back the latest
    # checkpoint reported below: continue from its weights and step instead of step 0.
    # The LR schedule resumes at that step; Adam's moments and the EMA restart from the
    # saved weights.
    start_step, carried = 0, []
    resume = ray.train.get_checkpoint()
    if resume:
        resume_dir = Path(resume.to_directory())
        start_step = json.loads((resume_dir / "resume.json").read_text())["step"]
        carried = json.loads((resume_dir / "loss_curve.json").read_text())
        config["of3"]["experiment_settings"]["restart_checkpoint_path"] = str(resume_dir / "resume.pt")
        if rank == 0:
            print(f"Resuming from the checkpoint at step {start_step} of {config['max_steps']}", flush=True)

    class GpuSampler:
        """Background sampler for GPU occupancy, drained once per optimizer step.

        `nvmlDeviceGetUtilizationRates` only says whether *any* kernel was resident, so it
        reads high even when the GPU is running one small kernel at a time. The number that
        answers "are the SMs busy" is the per-process `smUtil` from
        `nvmlDeviceGetProcessUtilization`, so both are recorded and reported separately.
        """

        def __init__(self, hz: float = 10.0):
            # Everything here is inside the guard: this is telemetry, and no failure in
            # it is worth losing a training run over.
            self.samples, self.ok, self.stop = [], False, None
            try:
                import pynvml
                pynvml.nvmlInit()
                self.stop = threading.Event()
                self.nvml = pynvml
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
                self.pid, self.ok = os.getpid(), True
                threading.Thread(target=self._loop, args=(1.0 / hz,), daemon=True).start()
            except Exception as e:
                self.ok = False
                print(f"GPU sampling unavailable, training continues: {e!r}", flush=True)

        def _loop(self, period):
            last_ts = 0
            while not self.stop.wait(period):
                try:
                    gpu = self.nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                    sm = 0
                    for p in self.nvml.nvmlDeviceGetProcessUtilization(self.handle, last_ts):
                        if p.pid == self.pid:
                            sm = p.smUtil
                    last_ts = int(time.time() * 1e6) - int(2 * period * 1e6)
                    self.samples.append((gpu, sm))
                except Exception:
                    pass                      # a sampling gap is not worth failing over

        def drain(self) -> dict:
            try:
                s, self.samples = self.samples, []
                if not s:
                    return {}
                return {"gpu_util_pct": round(sum(g for g, _ in s) / len(s), 1),
                        "sm_active_pct": round(sum(m for _, m in s) / len(s), 1)}
            except Exception:
                return {}

    class ReportEveryStep(pl.Callback):
        """Average the loss over workers and hand it to Ray Train after each optimizer step."""

        def __init__(self):
            self.t0, self.history = time.time(), list(carried)
            self.gpu = GpuSampler()

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            logged = {k[len("train/"):]: float(v) for k, v in trainer.callback_metrics.items()
                      if k.startswith("train/") and v.numel() == 1}
            logged["loss"] = float(outputs["loss"] if isinstance(outputs, dict) else outputs)
            # Average the headline terms over workers (each worker trains on a different complex)
            vals = torch.tensor([[logged.get(k, 0.0), k in logged] for k in REPORTED], device=module.device)
            vals = trainer.strategy.reduce(vals, reduce_op="sum")
            averaged = {k: (v / n if n else float("nan")) for k, (v, n) in zip(REPORTED, vals.tolist())}
            now = time.time()
            metrics = {
                "step": trainer.global_step + start_step,
                **averaged,
                "lr": trainer.optimizers[0].param_groups[0]["lr"],
                "step_time_s": round(now - self.t0, 2),
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
                if torch.cuda.is_available() else 0.0,
                **self.gpu.drain(),
                "worker0_pdb_id": batch["pdb_id"][0],
            }
            self.t0 = now
            self.history.append(metrics)
            window = [m["loss"] for m in self.history[-10:]]
            metrics["loss_avg10"] = sum(window) / len(window)
            # Smoothed protein-ligand lDDT: single steps are noisy (one complex per worker,
            # random noise level), so Ray Tune's early stopping compares this instead.
            lddt = [m["lddt_inter_protein_ligand_step"] for m in self.history[-20:]
                    if m["lddt_inter_protein_ligand_step"] == m["lddt_inter_protein_ligand_step"]]
            metrics["lddt_avg20"] = sum(lddt) / len(lddt) if lddt else float("nan")
            if rank == 0:
                print(f"step {metrics['step']:>4}  loss {metrics['loss']:7.3f}  avg10 {metrics['loss_avg10']:7.3f}  "
                      f"distogram {metrics['distogram_loss']:6.3f}  lr {metrics['lr']:.1e}  "
                      f"{metrics['step_time_s']:5.1f} s  {metrics['peak_mem_gb']:.1f} GB", flush=True)
            step = metrics["step"]
            if step == config["fail_at_step"] and not resume and rank == 0:
                raise RuntimeError(f"--fail-at-step {step}: simulated failure, to test resuming")
            every = config["checkpoint_every"]
            if not every or step % every or step >= config["max_steps"]:
                ray.train.report(metrics)
                return
            # Resume point: the raw weights (not the export name, so an unfinished run is
            # never mistaken for a finished one), its step, and the curve so far.
            with tempfile.TemporaryDirectory() as tmp:
                checkpoint = None
                if rank == 0:
                    torch.save({k: v.detach().cpu() for k, v in module.model.state_dict().items()},
                               Path(tmp) / "resume.pt")
                    (Path(tmp) / "resume.json").write_text(json.dumps({"step": step}))
                    (Path(tmp) / "loss_curve.json").write_text(json.dumps(self.history))
                    checkpoint = ray.train.Checkpoint.from_directory(tmp)
                ray.train.report(metrics, checkpoint=checkpoint)

    report = ReportEveryStep()

    class RayTrainingRunner(TrainingExperimentRunner):
        """OF3's training runner with the Trainer built for Ray Train workers."""

        @property
        def strategy(self):
            # Per-sample gradient clipping in OF3 requires a DDP strategy, even on one worker
            return RayDDPStrategy(find_unused_parameters=False)

        @property
        def callbacks(self):
            # No ModelCheckpoint: it writes a multi-GB file per epoch. ReportEveryStep
            # reports a resume checkpoint every --checkpoint-every steps and the final
            # weights at the end instead.
            return [RankSpecificSeedCallback(base_seed=self.seed), report]

        @property
        def trainer(self):
            if not hasattr(self, "_trainer"):
                args = self.pl_trainer_args.model_dump(
                    exclude={"deepspeed_config_path", "distributed_timeout", "mpi_plugin",
                             "devices", "num_nodes", "enable_checkpointing"})
                args.update(devices="auto", strategy=self.strategy, plugins=[RayLightningEnvironment()],
                            callbacks=self.callbacks, logger=CSVLogger(self.log_dir), enable_checkpointing=False,
                            enable_progress_bar=False, max_steps=config["max_steps"] - start_step,
                            default_root_dir=self.output_dir)
                self._trainer = prepare_trainer(pl.Trainer(**args))
            return self._trainer

    runner = RayTrainingRunner(TrainingExperimentConfig.model_validate(config["of3"]))
    runner.setup()      # builds the model and loads the starting (or resumed) weights
    if start_step:
        # The scheduler is built in trainer.fit; it then continues from start_step
        runner.lightning_module.resume_last_lr_step(start_step - 1)
    runner.run()        # trainer.fit with OF3's DataModule

    # Final weights: plain OF3 state dict (EMA or raw), reported to Ray as a checkpoint
    module = runner.lightning_module
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = None
        if rank == 0:
            weights = module.ema.state_dict()["params"] if config["export_ema"] else module.model.state_dict()
            torch.save({k: v.detach().cpu() for k, v in weights.items()}, Path(tmp) / "of3_ev71_ft.pt")
            (Path(tmp) / "loss_curve.json").write_text(json.dumps(report.history, indent=1))
            checkpoint = ray.train.Checkpoint.from_directory(tmp)
        last = {k: v for k, v in report.history[-1].items() if k != "worker0_pdb_id"} if report.history else {}
        ray.train.report({**last, "step": module.global_step + start_step, "final": True}, checkpoint=checkpoint)


def make_trainer(args, callbacks=()) -> TorchTrainer:
    """The TorchTrainer for one fine-tuning run. tune_finetune.py builds one per trial."""
    of3_config = build_config(args)
    TrainingExperimentConfig.model_validate(of3_config)   # fail fast, before any GPU starts
    scaling = ScalingConfig(
        num_workers=args.num_workers, use_gpu=True, accelerator_type=args.accelerator,
        # Featurization runs on CPU next to each GPU and is the usual reason an OF3 step
        # stalls. Reserve a core per loader process plus one for the trainer, always
        # leaving a couple free so Ray's own actors still get scheduled: 7 of 8 vCPU on a
        # g6.2xlarge, 13 of 16 on a g6.4xlarge.
        resources_per_worker={"CPU": min(args.loader_workers + 1, args.cpus_per_worker)})
    return TorchTrainer(
        train_loop_per_worker,
        train_loop_config={"of3": of3_config, "max_steps": args.max_steps, "export_ema": args.export_ema,
                           "checkpoint_every": args.checkpoint_every, "fail_at_step": args.fail_at_step},
        scaling_config=scaling,
        run_config=RunConfig(name=args.name, storage_path=str(args.storage), callbacks=list(callbacks),
                             # Spot capacity makes worker preemption normal rather than
                             # exceptional, and the default (0) kills the run on the first one.
                             failure_config=FailureConfig(max_failures=args.max_failures),
                             checkpoint_config=CheckpointConfig(num_to_keep=args.keep_checkpoints or None)))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-workers", type=int, default=2, help="GPU workers (one GPU each)")
    ap.add_argument("--max-steps", type=int, default=150, help="optimizer steps (one complex per worker per step)")
    ap.add_argument("--lr", type=float, default=1e-4, help="peak learning rate after warmup")
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--lr-final-frac", type=float, default=0.1,
                    help="learning rate at the last step, as a fraction of --lr")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="complexes per GPU per step. OF3 only supports 1: its per-sample "
                         "gradient clipping asserts on it (of3_all_atom/runner.py). To give "
                         "a step more work use --diffusion-samples; to give it more "
                         "complexes use --num-workers")
    ap.add_argument("--diffusion-samples", type=int, default=8, help="noise levels per complex per step")
    ap.add_argument("--accelerator", default="L4", help="GPU type for the workers")
    ap.add_argument("--loader-workers", type=int, default=4, help="featurization processes per GPU")
    ap.add_argument("--cpus-per-worker", type=int, default=7,
                    help="cap on CPUs reserved per GPU worker (7 fits a g6.2xlarge, 13 a g6.4xlarge)")
    ap.add_argument("--max-failures", type=int, default=2,
                    help="worker-group retries, for spot preemption")
    ap.add_argument("--export-ema", action="store_true", help="export EMA weights instead of the raw weights")
    ap.add_argument("--checkpoint-every", type=int, default=100,
                    help="steps between resume checkpoints (0: only the final weights). After a "
                         "restart, training continues from the latest one instead of step 0")
    ap.add_argument("--keep-checkpoints", type=int, default=1,
                    help="checkpoints kept in the run directory (0: all of them, to benchmark "
                         "accuracy against training steps; each is a 2.3 GB resume.pt)")
    ap.add_argument("--fail-at-step", type=int, default=-1,
                    help="testing: raise once at this step, to check that the run resumes")
    ap.add_argument("--data", type=Path, default=DATA, help="output of scripts/prepare_finetune.py")
    ap.add_argument("--base", default="ob0", choices=list(BASES),
                    help="starting weights: ob0 = OpenBind-0, p2 = of3-p2-145k (trains in OpenFold3 0.4.3)")
    ap.add_argument("--base-ckpt", type=Path, default=None, help="override the --base weights file")
    ap.add_argument("--storage", type=Path, default=SHARED / "finetune" / "runs")
    ap.add_argument("--name", default=None, help="run name (default: see run_name())")
    return ap


def base_ckpt(args) -> Path:
    return args.base_ckpt or BASES[args.base][0]


def check_inputs(args):
    for path in (args.data / "training_cache.json", base_ckpt(args)):
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run scripts/prepare_finetune.py / scripts/setup_data.py first.")
    # OF3 clips gradients per sample, which needs one complex per GPU per step, so it
    # asserts on anything else. Catch it here rather than after a cluster has started.
    if args.batch_size != 1:
        raise SystemExit(
            f"--batch-size {args.batch_size}: OpenFold3 supports only 1 complex per GPU per "
            "step (per-sample gradient clipping). Use --num-workers for more complexes per "
            "step, or --diffusion-samples for more work per complex.")


def fit(args) -> str:
    """Run the trainer and return the final checkpoint directory."""
    return make_trainer(args).fit().checkpoint.path


def main():
    args = parser().parse_args()
    py = BASES[args.base][1]
    if py and sys.executable != py:
        # The driver validates the OF3 config, so it must run the same OpenFold3 as the workers
        os.execv(py, [py, os.path.abspath(__file__), *sys.argv[1:]])
    args.name = args.name or run_name(args)
    check_inputs(args)
    print(f"run {args.name}: base {args.base} ({base_ckpt(args).name}), data {args.data.name}, "
          f"{args.num_workers} x {args.accelerator}, {args.max_steps} steps", flush=True)
    start = time.time()
    if py:
        # Ray starts workers with the cluster's own Python. Run the trainer inside a task
        # that uses the legacy one: the Ray Train controller and every worker it creates
        # inherit that task's runtime environment (and keep the job's working_dir).
        ray.init()
        env = {**ray.get_runtime_context().runtime_env, "py_executable": py}
        ckpt_path = ray.get(ray.remote(num_cpus=0)(fit).options(runtime_env=env).remote(args))
    else:
        ckpt_path = fit(args)

    ckpt_dir = Path(ckpt_path)
    curve = json.loads((ckpt_dir / "loss_curve.json").read_text())
    def mean(rows, key):
        return sum(r[key] for r in rows) / len(rows)

    first, last = curve[:10], curve[-10:]
    print(f"\nDone in {(time.time() - start) / 60:.1f} min: {len(curve)} steps on {args.num_workers} workers")
    for key in REPORTED:
        print(f"  {key:<30} first 10 steps {mean(first, key):.3f} -> last 10 steps {mean(last, key):.3f}")
    print(f"  median step time {sorted(r['step_time_s'] for r in curve)[len(curve) // 2]:.1f} s, "
          f"peak GPU memory {max(r['peak_mem_gb'] for r in curve):.1f} GB")
    print(f"  weights: {ckpt_dir / 'of3_ev71_ft.pt'}")


if __name__ == "__main__":
    main()
