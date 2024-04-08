# LL4MA Utilities

This repo contains several Python utility modules to cut down on code that needs to be copy-pasted between different packages.

---

## Modules

The following utility modules are currently offered:

| Module Name  | Description |
|--------------|-------------|
| `data_util`  | Data processing utilities (e.g. scale data, rotation conversions, image resizing, etc.) |
| `file_util`  | File processing utilities (e.g. load/save file types like yaml or pickle, directory ops, etc.) |
| `func_util`  | Function utilities (e.g. validating function inputs, convert data types, etc.) |
| `manip_util` | Utilities to aid manipulation planning/learning (e.g. compute poses w.r.t. bounding box) |
| `math_util`  | Math utilities (e.g. rotation conversions, quaternion operations, etc.) |
| `ros_util`   | ROS utilities (e.g. data/msg conversions, convenience functions, etc.) |
| `torch_util` | PyTorch utilities (e.g. moving models and data to device, change model mode, etc.) |
| `ui_util`    | UI command line utilities (e.g. print colored messages, query user, etc.) |
| `viz_util`   | Visualization utilities (e.g. randomizing colors) |

---

## Usage

Here is an example of how to import the modules in your package:
```python
from ll4ma_util import file_util

filenames = file_util.list_dir('/home/user/path/to/files', extension='.yaml')
```