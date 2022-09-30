# deep_tool
深度学习工具脚本（含自己编写和整理），基本都不能直接运行，只是为了存储API

## 目录
- 借用pycharm的工作区，把.pkl可视化
- [mmaction项目的，基于骨骼的视频可视化](mmaction/vis_skeleton.py)
- [mmaction项目的，基于骨骼的视频【批量】可视化](vis_skeleton_bupt8.py)

### 借用pycharm的工作区，把.pkl可视化
```python
import os
import pickle
import pandas as pd

path='name.pkl'
tool=pd.read_pickle(path)

print(tool.shape)
```
### mmaction项目的，基于骨骼的视频可视化
[vis_skeleton.py](mmaction/vis_skeleton.py)
代码整理+可真实运行，支持gym和ntu两个数据集，每次需要把另一个注释掉

### mmaction项目的，基于骨骼的视频【批量】可视化
[vis_skeleton_bupt8.py](vis_skeleton_bupt8.py)
遇到数据集的许多问题，解决方案目前有：
- 注释帧长度相同的assert
- 针对视频和.pkl包含的数据，不完全相同的问题，设置else分支
- 将问题视频录入error_list，先跳过不处理，运行结束后在命令行输出，推荐开tmux看下
