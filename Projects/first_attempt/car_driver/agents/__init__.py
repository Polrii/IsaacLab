# This file is required so python detects that there are files in this agents directory.

"""
If this file is not here, this error may occur:

Traceback (most recent call last):
  File "C:/Users/mcpek/IsaacLab/Projects/first_attempt/car_train.py", line 199, in <module>
    main()
  File "c:/users/mcpek/isaaclab/source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py", line 81, in wrapper
    env_cfg, agent_cfg = register_task_to_hydra(task_name, agent_cfg_entry_point)
  File "c:/users/mcpek/isaaclab/source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py", line 45, in register_task_to_hydra
    agent_cfg = load_cfg_from_registry(task_name, agent_cfg_entry_point)
  File "c:/users/mcpek/isaaclab/source/isaaclab_tasks/isaaclab_tasks/utils/parse_cfg.py", line 70, in load_cfg_from_registry
    mod_path = os.path.dirname(importlib.import_module(mod_name).__file__)
  File "C:/Users/mcpek/miniconda3/envs/env_isaaclab/lib/ntpath.py", line 249, in dirname
    return split(p)[0]
  File "C:/Users/mcpek/miniconda3/envs/env_isaaclab/lib/ntpath.py", line 211, in split
    p = os.fspath(p)
TypeError: expected str, bytes or os.PathLike object, not NoneType

"""