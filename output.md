```
tests/test_train.py::test_train_ddp_sim 
-------------------------------- live log call ---------------------------------
WARNING  src.utils.instantiators:pylogger.py:46 [rank: 0] No logger configs found! Skipping...
INFO     lightning.pytorch.utilities.rank_zero:setup.py:164 GPU available: False, used: False
INFO     lightning.pytorch.utilities.rank_zero:setup.py:167 TPU available: False, using: 0 TPU cores
INFO     lightning.pytorch.utilities.rank_zero:logger_connector.py:91 💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
INFO     lightning.fabric.utilities.distributed:distributed.py:282 Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/2
INFO     lightning.fabric.strategies.launchers.subprocess_script:subprocess_script.py:224 [rank: 1] Child process with PID 2977 terminated with code 4. Forcefully terminating all other processes to avoid zombies 🧟
/home/runner/work/_temp/b000d8bf-7776-4658-84e0-83f1f9f37dc2.sh: line 1:  2504 Killed                  pytest --cov src
Error: Process completed with exit code 137.
```