```shell
============================= test session starts ==============================
platform linux -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/runner/work/torch-cv-foundation-lightning/torch-cv-foundation-lightning
configfile: pyproject.toml
testpaths: tests/
plugins: hydra-core-1.3.5, cov-7.1.0
collected 16 items

tests/test_configs.py::test_train_config 
-------------------------------- live log call ---------------------------------
INFO     lightning.pytorch.utilities.rank_zero:setup.py:164 GPU available: False, used: False
INFO     lightning.pytorch.utilities.rank_zero:setup.py:167 TPU available: False, using: 0 TPU cores
INFO     lightning.pytorch.utilities.rank_zero:logger_connector.py:91 💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
PASSED                                                                   [  6%]
tests/test_configs.py::test_eval_config 
-------------------------------- live log call ---------------------------------
INFO     lightning.pytorch.utilities.rank_zero:setup.py:164 GPU available: False, used: False
INFO     lightning.pytorch.utilities.rank_zero:setup.py:167 TPU available: False, using: 0 TPU cores
INFO     lightning.pytorch.utilities.rank_zero:logger_connector.py:91 💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
PASSED                                                                   [ 12%]
tests/test_datamodules.py::test_mnist_datamodule[32] PASSED              [ 18%]
tests/test_datamodules.py::test_mnist_datamodule[128] PASSED             [ 25%]
tests/test_eval.py::test_train_eval 
-------------------------------- live log call ---------------------------------
WARNING  src.utils.instantiators:pylogger.py:46 [rank: 0] No logger configs found! Skipping...
INFO     lightning.pytorch.utilities.rank_zero:callback_connector.py:122 Trainer already configured with model summary callbacks: [<class 'lightning.pytorch.callbacks.rich_model_summary.RichModelSummary'>]. Skipping setting a default `ModelSummary` callback.
INFO     lightning.pytorch.utilities.rank_zero:setup.py:164 GPU available: False, used: False
INFO     lightning.pytorch.utilities.rank_zero:setup.py:167 TPU available: False, using: 0 TPU cores
INFO     lightning.pytorch.utilities.rank_zero:logger_connector.py:91 💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
INFO     lightning.pytorch.utilities.rank_zero:fit_loop.py:192 `Trainer.fit` stopped: `max_epochs=1` reached.
INFO     lightning.pytorch.utilities.rank_zero:checkpoint_connector.py:81 Restoring states from the checkpoint path at /tmp/pytest-of-runner/pytest-0/test_train_eval0/checkpoints/epoch_000.ckpt
INFO     lightning.pytorch.utilities.rank_zero:checkpoint_connector.py:224 Loaded model weights from the checkpoint at /tmp/pytest-of-runner/pytest-0/test_train_eval0/checkpoints/epoch_000.ckpt
WARNING  src.utils.instantiators:pylogger.py:46 [rank: 0] No logger configs found! Skipping...
INFO     lightning.pytorch.utilities.rank_zero:setup.py:164 GPU available: False, used: False
INFO     lightning.pytorch.utilities.rank_zero:setup.py:167 TPU available: False, using: 0 TPU cores
INFO     lightning.pytorch.utilities.rank_zero:logger_connector.py:91 💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
INFO     lightning.pytorch.utilities.rank_zero:callback_connector.py:190 💡 Tip: For seamless cloud uploads and versioning, try installing [litmodels](https://pypi.org/project/litmodels/) to enable LitModelCheckpoint, which syncs automatically with the Lightning model registry.
INFO     lightning.pytorch.utilities.rank_zero:checkpoint_connector.py:81 Restoring states from the checkpoint path at /tmp/pytest-of-runner/pytest-0/test_train_eval0/checkpoints/last.ckpt
INFO     lightning.pytorch.utilities.rank_zero:checkpoint_connector.py:224 Loaded model weights from the checkpoint at /tmp/pytest-of-runner/pytest-0/test_train_eval0/checkpoints/last.ckpt
PASSED                                                                   [ 31%]
tests/test_sweeps.py::test_experiments PASSED                            [ 37%]
tests/test_sweeps.py::test_hydra_sweep PASSED                            [ 43%]
tests/test_sweeps.py::test_hydra_sweep_ddp_sim FAILED                    [ 50%]
tests/test_sweeps.py::test_optuna_sweep PASSED                           [ 56%]
tests/test_sweeps.py::test_optuna_sweep_ddp_sim_wandb SKIPPED (Requi...) [ 62%]
tests/test_train.py::test_train_fast_dev_run 
-------------------------------- live log call ---------------------------------
WARNING  src.utils.instantiators:pylogger.py:46 [rank: 0] No logger configs found! Skipping...
INFO     lightning.pytorch.utilities.rank_zero:callback_connector.py:122 Trainer already configured with model summary callbacks: [<class 'lightning.pytorch.callbacks.rich_model_summary.RichModelSummary'>]. Skipping setting a default `ModelSummary` callback.
INFO     lightning.pytorch.utilities.rank_zero:setup.py:164 GPU available: False, used: False
INFO     lightning.pytorch.utilities.rank_zero:setup.py:167 TPU available: False, using: 0 TPU cores
INFO     lightning.pytorch.utilities.rank_zero:logger_connector.py:91 💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
INFO     lightning.pytorch.utilities.rank_zero:setup.py:75 Running in `fast_dev_run` mode: will run the requested loop using 1 batch(es). Logging and checkpointing is suppressed.
INFO     lightning.pytorch.utilities.rank_zero:fit_loop.py:182 `Trainer.fit` stopped: `max_steps=1` reached.
WARNING  src.train:pylogger.py:46 [rank: 0] Best ckpt not found! Using current weights for testing...
PASSED                                                                   [ 68%]
tests/test_train.py::test_train_fast_dev_run_gpu SKIPPED (Requires: ...) [ 75%]
tests/test_train.py::test_train_epoch_gpu_amp SKIPPED (Requires: [GP...) [ 81%]
tests/test_train.py::test_train_epoch_double_val_loop 
-------------------------------- live log call ---------------------------------
WARNING  src.utils.instantiators:pylogger.py:46 [rank: 0] No logger configs found! Skipping...
INFO     lightning.pytorch.utilities.rank_zero:callback_connector.py:122 Trainer already configured with model summary callbacks: [<class 'lightning.pytorch.callbacks.rich_model_summary.RichModelSummary'>]. Skipping setting a default `ModelSummary` callback.
INFO     lightning.pytorch.utilities.rank_zero:setup.py:164 GPU available: False, used: False
INFO     lightning.pytorch.utilities.rank_zero:setup.py:167 TPU available: False, using: 0 TPU cores
INFO     lightning.pytorch.utilities.rank_zero:logger_connector.py:91 💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
INFO     lightning.pytorch.utilities.rank_zero:fit_loop.py:192 `Trainer.fit` stopped: `max_epochs=1` reached.
INFO     lightning.pytorch.utilities.rank_zero:checkpoint_connector.py:81 Restoring states from the checkpoint path at /tmp/pytest-of-runner/pytest-0/test_train_epoch_double_val_lo0/checkpoints/epoch_000.ckpt
INFO     lightning.pytorch.utilities.rank_zero:checkpoint_connector.py:224 Loaded model weights from the checkpoint at /tmp/pytest-of-runner/pytest-0/test_train_epoch_double_val_lo0/checkpoints/epoch_000.ckpt
PASSED                                                                   [ 87%]
tests/test_train.py::test_train_ddp_sim 
-------------------------------- live log call ---------------------------------
WARNING  src.utils.instantiators:pylogger.py:46 [rank: 0] No logger configs found! Skipping...
INFO     lightning.pytorch.utilities.rank_zero:callback_connector.py:122 Trainer already configured with model summary callbacks: [<class 'lightning.pytorch.callbacks.rich_model_summary.RichModelSummary'>]. Skipping setting a default `ModelSummary` callback.
INFO     lightning.pytorch.utilities.rank_zero:setup.py:164 GPU available: False, used: False
INFO     lightning.pytorch.utilities.rank_zero:setup.py:167 TPU available: False, using: 0 TPU cores
INFO     lightning.pytorch.utilities.rank_zero:logger_connector.py:91 💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
WARNING  torch.multiprocessing.spawn:spawn.py:165 Terminating process 3009 via signal SIGTERM
ERROR    src.utils.utils:pylogger.py:46 [rank: 0] 
Traceback (most recent call last):
  File "/home/runner/work/torch-cv-foundation-lightning/torch-cv-foundation-lightning/src/utils/utils.py", line 68, in wrap
    metric_dict, object_dict = task_func(cfg=cfg)
                               ^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/torch-cv-foundation-lightning/torch-cv-foundation-lightning/src/train.py", line 87, in train
    trainer.fit(
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/lightning/pytorch/trainer/trainer.py", line 584, in fit
    call._call_and_handle_interrupt(
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/lightning/pytorch/trainer/call.py", line 48, in _call_and_handle_interrupt
    return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/lightning/pytorch/strategies/launchers/multiprocessing.py", line 144, in launch
    while not process_context.join():
              ^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/torch/multiprocessing/spawn.py", line 200, in join
    raise ProcessExitedException(
torch.multiprocessing.spawn.ProcessExitedException: process 1 terminated with exit code 1
FAILED                                                                   [ 93%]
tests/test_train.py::test_train_resume 
-------------------------------- live log call ---------------------------------
WARNING  src.utils.instantiators:pylogger.py:46 [rank: 0] No logger configs found! Skipping...
INFO     lightning.pytorch.utilities.rank_zero:callback_connector.py:122 Trainer already configured with model summary callbacks: [<class 'lightning.pytorch.callbacks.rich_model_summary.RichModelSummary'>]. Skipping setting a default `ModelSummary` callback.
INFO     lightning.pytorch.utilities.rank_zero:setup.py:164 GPU available: False, used: False
INFO     lightning.pytorch.utilities.rank_zero:setup.py:167 TPU available: False, using: 0 TPU cores
INFO     lightning.pytorch.utilities.rank_zero:logger_connector.py:91 💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
INFO     lightning.pytorch.utilities.rank_zero:fit_loop.py:192 `Trainer.fit` stopped: `max_epochs=1` reached.
INFO     lightning.pytorch.utilities.rank_zero:checkpoint_connector.py:81 Restoring states from the checkpoint path at /tmp/pytest-of-runner/pytest-0/test_train_resume0/checkpoints/epoch_000.ckpt
INFO     lightning.pytorch.utilities.rank_zero:checkpoint_connector.py:224 Loaded model weights from the checkpoint at /tmp/pytest-of-runner/pytest-0/test_train_resume0/checkpoints/epoch_000.ckpt
WARNING  src.utils.instantiators:pylogger.py:46 [rank: 0] No logger configs found! Skipping...
INFO     lightning.pytorch.utilities.rank_zero:callback_connector.py:122 Trainer already configured with model summary callbacks: [<class 'lightning.pytorch.callbacks.rich_model_summary.RichModelSummary'>]. Skipping setting a default `ModelSummary` callback.
INFO     lightning.pytorch.utilities.rank_zero:setup.py:164 GPU available: False, used: False
INFO     lightning.pytorch.utilities.rank_zero:setup.py:167 TPU available: False, using: 0 TPU cores
INFO     lightning.pytorch.utilities.rank_zero:logger_connector.py:91 💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
INFO     lightning.pytorch.utilities.rank_zero:checkpoint_connector.py:81 Restoring states from the checkpoint path at /tmp/pytest-of-runner/pytest-0/test_train_resume0/checkpoints/last.ckpt
INFO     lightning.pytorch.utilities.rank_zero:checkpoint_connector.py:224 Restored all states from the checkpoint at /tmp/pytest-of-runner/pytest-0/test_train_resume0/checkpoints/last.ckpt
INFO     lightning.pytorch.utilities.rank_zero:fit_loop.py:192 `Trainer.fit` stopped: `max_epochs=2` reached.
INFO     lightning.pytorch.utilities.rank_zero:checkpoint_connector.py:81 Restoring states from the checkpoint path at /tmp/pytest-of-runner/pytest-0/test_train_resume0/checkpoints/epoch_001.ckpt
INFO     lightning.pytorch.utilities.rank_zero:checkpoint_connector.py:224 Loaded model weights from the checkpoint at /tmp/pytest-of-runner/pytest-0/test_train_resume0/checkpoints/epoch_001.ckpt
PASSED                                                                   [100%]

=================================== FAILURES ===================================
___________________________ test_hydra_sweep_ddp_sim ___________________________

tmp_path = PosixPath('/tmp/pytest-of-runner/pytest-0/test_hydra_sweep_ddp_sim0')

    @RunIf(sh=True)
    @pytest.mark.slow
    def test_hydra_sweep_ddp_sim(tmp_path: Path) -> None:
        """Test default hydra sweep with ddp sim.
    
        :param tmp_path: The temporary logging path.
        """
        command = [
            startfile,
            "-m",
            "hydra.sweep.dir=" + str(tmp_path),
            "trainer=ddp_sim",
            "trainer.max_epochs=3",
            "+trainer.limit_train_batches=0.01",
            "+trainer.limit_val_batches=0.1",
            "+trainer.limit_test_batches=0.1",
            "model.optimizer.lr=0.005,0.01,0.02",
        ] + overrides
>       run_sh_command(command)

tests/test_sweeps.py:65: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

command = ['src/train.py', '-m', 'hydra.sweep.dir=/tmp/pytest-of-runner/pytest-0/test_hydra_sweep_ddp_sim0', 'trainer=ddp_sim', 'trainer.max_epochs=3', '+trainer.limit_train_batches=0.01', ...]

    def run_sh_command(command: List[str]) -> None:
        """Default method for executing shell commands with `pytest` and `sh` package.
    
        :param command: A list of shell commands as strings.
        """
        msg = None
        try:
            sh.python(command)
        except sh.ErrorReturnCode as e:
            msg = e.stderr.decode()
        if msg:
>           pytest.fail(msg)
E           Failed: Trainer already configured with model summary callbacks: [<class 'lightning.pytorch.callbacks.rich_model_summary.RichModelSummary'>]. Skipping setting a default `ModelSummary` callback.
E           GPU available: False, used: False
E           TPU available: False, using: 0 TPU cores
E           💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
E           Traceback (most recent call last):
E             File "<string>", line 1, in <module>
E             File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 122, in spawn_main
E           Traceback (most recent call last):
E             File "<string>", line 1, in <module>
E               exitcode = _main(fd, parent_sentinel)
E             File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 122, in spawn_main
E                          ^^^^^^^^^^^^^^^^^^^^^^^^^^
E             File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 132, in _main
E               self = reduction.pickle.load(from_parent)
E                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E             File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/torch/multiprocessing/reductions.py", line 543, in rebuild_storage_fd
E               exitcode = _main(fd, parent_sentinel)
E                          ^^^^^^^^^^^^^^^^^^^^^^^^^^
E             File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 132, in _main
E               storage = cls._new_shared_fd_cpu(fd, size)
E                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E           RuntimeError: unable to resize file <filename not specified> to the right size: Invalid argument (22)
E               self = reduction.pickle.load(from_parent)
E                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E             File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/torch/multiprocessing/reductions.py", line 543, in rebuild_storage_fd
E               storage = cls._new_shared_fd_cpu(fd, size)
E                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E           RuntimeError: unable to resize file <filename not specified> to the right size: Invalid argument (22)
E           W0821 08:08:07.455000 2542 site-packages/torch/multiprocessing/spawn.py:165] Terminating process 2556 via signal SIGTERM
E           Trainer already configured with model summary callbacks: [<class 'lightning.pytorch.callbacks.rich_model_summary.RichModelSummary'>]. Skipping setting a default `ModelSummary` callback.
E           GPU available: False, used: False
E           TPU available: False, using: 0 TPU cores
E           💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
E           Traceback (most recent call last):
E             File "<string>", line 1, in <module>
E             File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 122, in spawn_main
E               exitcode = _main(fd, parent_sentinel)
E                          ^^^^^^^^^^^^^^^^^^^^^^^^^^
E             File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 132, in _main
E               self = reduction.pickle.load(from_parent)
E                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E             File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/torch/multiprocessing/reductions.py", line 543, in rebuild_storage_fd
E               storage = cls._new_shared_fd_cpu(fd, size)
E                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E           RuntimeError: unable to resize file <filename not specified> to the right size: Invalid argument (22)
E           Traceback (most recent call last):
E             File "<string>", line 1, in <module>
E             File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 122, in spawn_main
E               exitcode = _main(fd, parent_sentinel)
E                          ^^^^^^^^^^^^^^^^^^^^^^^^^^
E             File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 132, in _main
E               self = reduction.pickle.load(from_parent)
E                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E             File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/torch/multiprocessing/reductions.py", line 543, in rebuild_storage_fd
E               storage = cls._new_shared_fd_cpu(fd, size)
E                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E           RuntimeError: unable to resize file <filename not specified> to the right size: Invalid argument (22)
E           W0821 08:08:12.773000 2542 site-packages/torch/multiprocessing/spawn.py:165] Terminating process 2573 via signal SIGTERM
E           Trainer already configured with model summary callbacks: [<class 'lightning.pytorch.callbacks.rich_model_summary.RichModelSummary'>]. Skipping setting a default `ModelSummary` callback.
E           GPU available: False, used: False
E           TPU available: False, using: 0 TPU cores
E           💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
E           Traceback (most recent call last):
E             File "<string>", line 1, in <module>
E             File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 122, in spawn_main
E               exitcode = _main(fd, parent_sentinel)
E                          ^^^^^^^^^^^^^^^^^^^^^^^^^^
E             File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 132, in _main
E               self = reduction.pickle.load(from_parent)
E                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E             File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/torch/multiprocessing/reductions.py", line 543, in rebuild_storage_fd
E               storage = cls._new_shared_fd_cpu(fd, size)
E                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E           RuntimeError: unable to resize file <filename not specified> to the right size: Invalid argument (22)
E           Traceback (most recent call last):
E             File "<string>", line 1, in <module>
E             File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 122, in spawn_main
E               exitcode = _main(fd, parent_sentinel)
E                          ^^^^^^^^^^^^^^^^^^^^^^^^^^
E             File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 132, in _main
E               self = reduction.pickle.load(from_parent)
E                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E             File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/torch/multiprocessing/reductions.py", line 543, in rebuild_storage_fd
E               storage = cls._new_shared_fd_cpu(fd, size)
E                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E           RuntimeError: unable to resize file <filename not specified> to the right size: Invalid argument (22)
E           W0821 08:08:18.251000 2542 site-packages/torch/multiprocessing/spawn.py:165] Terminating process 2588 via signal SIGTERM
E           Error executing job with overrides: ['trainer=ddp_sim', 'trainer.max_epochs=3', '+trainer.limit_train_batches=0.01', '+trainer.limit_val_batches=0.1', '+trainer.limit_test_batches=0.1', 'model.optimizer.lr=0.005', 'logger=[]']
E           Traceback (most recent call last):
E             File "/home/runner/work/torch-cv-foundation-lightning/torch-cv-foundation-lightning/src/train.py", line 122, in main
E               metric_dict, _ = train(cfg)
E                                ^^^^^^^^^^
E             File "/home/runner/work/torch-cv-foundation-lightning/torch-cv-foundation-lightning/src/utils/utils.py", line 78, in wrap
E               raise ex
E             File "/home/runner/work/torch-cv-foundation-lightning/torch-cv-foundation-lightning/src/utils/utils.py", line 68, in wrap
E               metric_dict, object_dict = task_func(cfg=cfg)
E                                          ^^^^^^^^^^^^^^^^^^
E             File "/home/runner/work/torch-cv-foundation-lightning/torch-cv-foundation-lightning/src/train.py", line 87, in train
E               trainer.fit(
E             File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/lightning/pytorch/trainer/trainer.py", line 584, in fit
E               call._call_and_handle_interrupt(
E             File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/lightning/pytorch/trainer/call.py", line 48, in _call_and_handle_interrupt
E               return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
E                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E             File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/lightning/pytorch/strategies/launchers/multiprocessing.py", line 144, in launch
E               while not process_context.join():
E                         ^^^^^^^^^^^^^^^^^^^^^^
E             File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/torch/multiprocessing/spawn.py", line 200, in join
E               raise ProcessExitedException(
E           torch.multiprocessing.spawn.ProcessExitedException: process 0 terminated with exit code 1
E           
E           Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

tests/helpers/run_sh_command.py:22: Failed
______________________________ test_train_ddp_sim ______________________________

cfg_train = {'hydra': {'run': {'dir': '${paths.log_dir}/${task_name}/runs/${now:%Y-%m-%d}_${now:%H-%M-%S}'}, 'sweep': {'dir': '${p...work_dir': '${hydra:runtime.cwd}'}, 'extras': {'ignore_warnings': False, 'enforce_tags': False, 'print_config': False}}

    @pytest.mark.slow
    def test_train_ddp_sim(cfg_train: DictConfig) -> None:
        """Simulate DDP (Distributed Data Parallel) on 2 CPU processes.
    
        :param cfg_train: A DictConfig containing a valid training configuration.
        """
        HydraConfig().set_config(cfg_train)
        with open_dict(cfg_train):
            cfg_train.trainer.max_epochs = 2
            cfg_train.trainer.accelerator = "cpu"
            cfg_train.trainer.devices = 2
            cfg_train.trainer.strategy = "ddp_spawn"
>       train(cfg_train)

tests/test_train.py:77: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
src/utils/utils.py:78: in wrap
    raise ex
src/utils/utils.py:68: in wrap
    metric_dict, object_dict = task_func(cfg=cfg)
                               ^^^^^^^^^^^^^^^^^^
src/train.py:87: in train
    trainer.fit(
/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/lightning/pytorch/trainer/trainer.py:584: in fit
    call._call_and_handle_interrupt(
/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/lightning/pytorch/trainer/call.py:48: in _call_and_handle_interrupt
    return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/lightning/pytorch/strategies/launchers/multiprocessing.py:144: in launch
    while not process_context.join():
              ^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <torch.multiprocessing.spawn.ProcessContext object at 0x7ff0ca90f0b0>
timeout = None, grace_period = None

    def join(self, timeout: float | None = None, grace_period: float | None = None):
        r"""Join one or more processes within spawn context.
    
        Attempt to join one or more processes in this spawn context.
        If one of them exited with a non-zero exit status, this function
        kills the remaining processes (optionally with a grace period)
        and raises an exception with the cause of the first process exiting.
    
        Returns ``True`` if all processes have been joined successfully,
        ``False`` if there are more processes that need to be joined.
    
        Args:
            timeout (float): Wait this long (in seconds) before giving up on waiting.
            grace_period (float): When any processes fail, wait this long (in seconds)
                for others to shutdown gracefully before terminating them. If they
                still don't exit, wait another grace period before killing them.
        """
        # Ensure this function can be called even when we're done.
        if len(self.sentinels) == 0:
            return True
    
        # Wait for any process to fail or all of them to succeed.
        ready = multiprocessing.connection.wait(
            self.sentinels.keys(),
            timeout=timeout,
        )
    
        error_index = None
        for sentinel in ready:
            index = self.sentinels.pop(sentinel)
            process = self.processes[index]
            process.join()
            if process.exitcode != 0:
                error_index = index
                break
    
        # Return if there was no error.
        if error_index is None:
            # Return whether or not all processes have been joined.
            return len(self.sentinels) == 0
        # An error occurred. Clean-up all processes before returning.
        # First, allow a grace period for processes to shutdown themselves.
        if grace_period is not None:
            self._join_procs_with_timeout(grace_period)
        # Then, terminate processes that are still alive. Try SIGTERM first.
        for process in self.processes:
            if process.is_alive():
                log.warning("Terminating process %s via signal SIGTERM", process.pid)
                process.terminate()
    
        # Try SIGKILL if the process isn't going down after another grace_period.
        # The reason is related to python signal handling is limited
        # to main thread and if that is in c/c++ land and stuck it won't
        # to handle it. We have seen processes getting stuck not handling
        # SIGTERM for the above reason.
        self._join_procs_with_timeout(30 if grace_period is None else grace_period)
        for process in self.processes:
            if process.is_alive():
                log.warning(
                    "Unable to shutdown process %s via SIGTERM , forcefully exiting via SIGKILL",
                    process.pid,
                )
                process.kill()
            process.join()
    
        # The file will only be created if the process crashed.
        failed_process = self.processes[error_index]
        if not os.access(self.error_files[error_index], os.R_OK):
            exitcode = self.processes[error_index].exitcode
            if exitcode < 0:
                try:
                    name = signal.Signals(-exitcode).name
                except ValueError:
                    name = f"<Unknown signal {-exitcode}>"
                raise ProcessExitedException(
                    f"process {error_index:d} terminated with signal {name}",
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode,
                    signal_name=name,
                )
            else:
>               raise ProcessExitedException(
                    f"process {error_index:d} terminated with exit code {exitcode:d}",
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode,
                )
E               torch.multiprocessing.spawn.ProcessExitedException: process 1 terminated with exit code 1

/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/torch/multiprocessing/spawn.py:200: ProcessExitedException
----------------------------- Captured stderr call -----------------------------
INFO: Trainer already configured with model summary callbacks: [<class 'lightning.pytorch.callbacks.rich_model_summary.RichModelSummary'>]. Skipping setting a default `ModelSummary` callback.
INFO: GPU available: False, used: False
INFO: TPU available: False, using: 0 TPU cores
INFO: 💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 132, in _main
    self = reduction.pickle.load(from_parent)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/torch/multiprocessing/reductions.py", line 543, in rebuild_storage_fd
    storage = cls._new_shared_fd_cpu(fd, size)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: unable to resize file <filename not specified> to the right size: Invalid argument (22)
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 132, in _main
    self = reduction.pickle.load(from_parent)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/torch/multiprocessing/reductions.py", line 543, in rebuild_storage_fd
    storage = cls._new_shared_fd_cpu(fd, size)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: unable to resize file <filename not specified> to the right size: Invalid argument (22)
W0821 08:08:40.502000 2363 site-packages/torch/multiprocessing/spawn.py:165] Terminating process 3009 via signal SIGTERM
------------------------------ Captured log call -------------------------------
WARNING  src.utils.instantiators:pylogger.py:46 [rank: 0] No logger configs found! Skipping...
INFO     lightning.pytorch.utilities.rank_zero:callback_connector.py:122 Trainer already configured with model summary callbacks: [<class 'lightning.pytorch.callbacks.rich_model_summary.RichModelSummary'>]. Skipping setting a default `ModelSummary` callback.
INFO     lightning.pytorch.utilities.rank_zero:setup.py:164 GPU available: False, used: False
INFO     lightning.pytorch.utilities.rank_zero:setup.py:167 TPU available: False, using: 0 TPU cores
INFO     lightning.pytorch.utilities.rank_zero:logger_connector.py:91 💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
WARNING  torch.multiprocessing.spawn:spawn.py:165 Terminating process 3009 via signal SIGTERM
ERROR    src.utils.utils:pylogger.py:46 [rank: 0] 
Traceback (most recent call last):
  File "/home/runner/work/torch-cv-foundation-lightning/torch-cv-foundation-lightning/src/utils/utils.py", line 68, in wrap
    metric_dict, object_dict = task_func(cfg=cfg)
                               ^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/torch-cv-foundation-lightning/torch-cv-foundation-lightning/src/train.py", line 87, in train
    trainer.fit(
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/lightning/pytorch/trainer/trainer.py", line 584, in fit
    call._call_and_handle_interrupt(
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/lightning/pytorch/trainer/call.py", line 48, in _call_and_handle_interrupt
    return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/lightning/pytorch/strategies/launchers/multiprocessing.py", line 144, in launch
    while not process_context.join():
              ^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/torch/multiprocessing/spawn.py", line 200, in join
    raise ProcessExitedException(
torch.multiprocessing.spawn.ProcessExitedException: process 1 terminated with exit code 1
=============================== warnings summary ===============================
tests/test_eval.py::test_train_eval
tests/test_train.py::test_train_fast_dev_run
tests/test_train.py::test_train_epoch_double_val_loop
tests/test_train.py::test_train_resume
  /opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/lightning/pytorch/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================================ tests coverage ================================
_______________ coverage: platform linux, python 3.12.14-final-0 _______________

Name                                  Stmts   Miss  Cover
---------------------------------------------------------
src/__init__.py                           0      0   100%
src/data/__init__.py                      3      0   100%
src/data/components/__init__.py           0      0   100%
src/data/mnist_datamodule.py            110     29    74%
src/data/utils.py                        21     12    43%
src/eval.py                              36      7    81%
src/infer.py                             16     16     0%
src/loss/__init__.py                      2      0   100%
src/loss/loss.py                        178    137    23%
src/loss/qr.py                           28     28     0%
src/metrics/__init__.py                   4      0   100%
src/metrics/bak/__init__.py               2      2     0%
src/metrics/bak/metrics.py              219    219     0%
src/metrics/general.py                  143     85    41%
src/metrics/metrics.py                  414    306    26%
src/metrics/metrics_dev.py              298    229    23%
src/models/__init__.py                    3      0   100%
src/models/components/__init__.py         2      0   100%
src/models/components/simple_net.py      25      5    80%
src/models/mnist_module.py               73      4    95%
src/optimizers/__init__.py                3      3     0%
src/optimizers/builder.py                99     99     0%
src/optimizers/schedulers.py             20     20     0%
src/train.py                             49      7    86%
src/utils/__init__.py                     6      0   100%
src/utils/instantiators.py               31     10    68%
src/utils/logging_utils.py               28     21    25%
src/utils/pylogger.py                    22      5    77%
src/utils/rich_utils.py                  46     31    33%
src/utils/utils.py                       43     24    44%
---------------------------------------------------------
TOTAL                                  1924   1299    32%
============================== slowest durations ===============================
21.97s call     tests/test_sweeps.py::test_hydra_sweep_ddp_sim
14.82s call     tests/test_sweeps.py::test_optuna_sweep
7.16s call     tests/test_sweeps.py::test_hydra_sweep
6.27s call     tests/test_sweeps.py::test_experiments
5.09s call     tests/test_train.py::test_train_ddp_sim
1.76s call     tests/test_eval.py::test_train_eval
1.54s call     tests/test_train.py::test_train_resume
0.99s call     tests/test_datamodules.py::test_mnist_datamodule[32]
0.91s call     tests/test_train.py::test_train_epoch_double_val_loop
0.31s setup    tests/test_configs.py::test_train_config
0.28s call     tests/test_train.py::test_train_fast_dev_run
0.18s setup    tests/test_configs.py::test_eval_config
0.16s call     tests/test_datamodules.py::test_mnist_datamodule[128]
0.08s call     tests/test_configs.py::test_train_config
0.04s call     tests/test_configs.py::test_eval_config
0.03s setup    tests/test_eval.py::test_train_eval
0.02s setup    tests/test_train.py::test_train_fast_dev_run
0.02s setup    tests/test_train.py::test_train_epoch_double_val_loop
0.02s setup    tests/test_train.py::test_train_ddp_sim
0.02s setup    tests/test_train.py::test_train_resume

(25 durations < 0.005s hidden.  Use -vv to show these durations.)
=========================== short test summary info ============================
FAILED tests/test_sweeps.py::test_hydra_sweep_ddp_sim - Failed: Trainer already configured with model summary callbacks: [<class 'lightning.pytorch.callbacks.rich_model_summary.RichModelSummary'>]. Skipping setting a default `ModelSummary` callback.
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 122, in spawn_main
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    exitcode = _main(fd, parent_sentinel)
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 122, in spawn_main
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 132, in _main
    self = reduction.pickle.load(from_parent)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/torch/multiprocessing/reductions.py", line 543, in rebuild_storage_fd
    exitcode = _main(fd, parent_sentinel)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 132, in _main
    storage = cls._new_shared_fd_cpu(fd, size)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: unable to resize file <filename not specified> to the right size: Invalid argument (22)
    self = reduction.pickle.load(from_parent)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/torch/multiprocessing/reductions.py", line 543, in rebuild_storage_fd
    storage = cls._new_shared_fd_cpu(fd, size)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: unable to resize file <filename not specified> to the right size: Invalid argument (22)
W0821 08:08:07.455000 2542 site-packages/torch/multiprocessing/spawn.py:165] Terminating process 2556 via signal SIGTERM
Trainer already configured with model summary callbacks: [<class 'lightning.pytorch.callbacks.rich_model_summary.RichModelSummary'>]. Skipping setting a default `ModelSummary` callback.
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 132, in _main
    self = reduction.pickle.load(from_parent)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/torch/multiprocessing/reductions.py", line 543, in rebuild_storage_fd
    storage = cls._new_shared_fd_cpu(fd, size)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: unable to resize file <filename not specified> to the right size: Invalid argument (22)
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 132, in _main
    self = reduction.pickle.load(from_parent)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/torch/multiprocessing/reductions.py", line 543, in rebuild_storage_fd
    storage = cls._new_shared_fd_cpu(fd, size)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: unable to resize file <filename not specified> to the right size: Invalid argument (22)
W0821 08:08:12.773000 2542 site-packages/torch/multiprocessing/spawn.py:165] Terminating process 2573 via signal SIGTERM
Trainer already configured with model summary callbacks: [<class 'lightning.pytorch.callbacks.rich_model_summary.RichModelSummary'>]. Skipping setting a default `ModelSummary` callback.
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 132, in _main
    self = reduction.pickle.load(from_parent)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/torch/multiprocessing/reductions.py", line 543, in rebuild_storage_fd
    storage = cls._new_shared_fd_cpu(fd, size)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: unable to resize file <filename not specified> to the right size: Invalid argument (22)
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/multiprocessing/spawn.py", line 132, in _main
    self = reduction.pickle.load(from_parent)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/torch/multiprocessing/reductions.py", line 543, in rebuild_storage_fd
    storage = cls._new_shared_fd_cpu(fd, size)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: unable to resize file <filename not specified> to the right size: Invalid argument (22)
W0821 08:08:18.251000 2542 site-packages/torch/multiprocessing/spawn.py:165] Terminating process 2588 via signal SIGTERM
Error executing job with overrides: ['trainer=ddp_sim', 'trainer.max_epochs=3', '+trainer.limit_train_batches=0.01', '+trainer.limit_val_batches=0.1', '+trainer.limit_test_batches=0.1', 'model.optimizer.lr=0.005', 'logger=[]']
Traceback (most recent call last):
  File "/home/runner/work/torch-cv-foundation-lightning/torch-cv-foundation-lightning/src/train.py", line 122, in main
    metric_dict, _ = train(cfg)
                     ^^^^^^^^^^
  File "/home/runner/work/torch-cv-foundation-lightning/torch-cv-foundation-lightning/src/utils/utils.py", line 78, in wrap
    raise ex
  File "/home/runner/work/torch-cv-foundation-lightning/torch-cv-foundation-lightning/src/utils/utils.py", line 68, in wrap
    metric_dict, object_dict = task_func(cfg=cfg)
                               ^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/torch-cv-foundation-lightning/torch-cv-foundation-lightning/src/train.py", line 87, in train
    trainer.fit(
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/lightning/pytorch/trainer/trainer.py", line 584, in fit
    call._call_and_handle_interrupt(
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/lightning/pytorch/trainer/call.py", line 48, in _call_and_handle_interrupt
    return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/lightning/pytorch/strategies/launchers/multiprocessing.py", line 144, in launch
    while not process_context.join():
              ^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/torch/multiprocessing/spawn.py", line 200, in join
    raise ProcessExitedException(
torch.multiprocessing.spawn.ProcessExitedException: process 0 terminated with exit code 1

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
FAILED tests/test_train.py::test_train_ddp_sim - torch.multiprocessing.spawn.ProcessExitedException: process 1 terminated with exit code 1
======== 2 failed, 11 passed, 3 skipped, 4 warnings in 70.93s (0:01:10) ========
Error: Process completed with exit code 1.
```