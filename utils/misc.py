"""
[杂项工具模块] misc.py

定位：
本模块为临时性代码容器，存放项目中无法明确分类的非核心功能代码。所有代码应保持低耦合、可迁移性。

主要包含：
1. 工具函数：
   - 文件/路径操作：`safe_mkdir`, `split_file_extension`
   - 调试工具：`print_data_shape`, `log_memory_usage`
   - 随机控制：`set_seed`, `disable_random`

2. 实验性代码
   - 验证性功能的临时实现（用后即删）
   - 废弃接口的兼容层（标记过期时间）

3. 第三方适配
   - 不同深度学习框架(PyTorch/TensorFlow)的转换工具
   - 数据格式转换（如 tensor_to_numpy)

维护原则：
⚠️ 禁止添加核心逻辑！所有代码应有明确注释和迁移计划。
🔄 每两个月检查一次，将成熟代码拆分到 utils/。
🗑️ 已废弃代码保留不超过两个版本周期。

使用示例：
    >>> from misc import set_seed
    >>> set_seed(42)  # 固定随机种子确保可复现性
"""

import os
import omegaconf
from omegaconf import OmegaConf

import torch
import torch.backends

import numpy as np
import random
from pathlib import Path


def omegaconf_to_dotdict(hparams):
    """
    将 OmegaConf 的嵌套配置对象(DictConfig/ListConfig)转换为展平的点分隔键字典。

    例如：
    输入 OmegaConf 配置：{"model": {"lr": 0.01, "name": "resnet"}}
    输出字典：{"model.lr": 0.01, "model.name": "resnet"}

    Args:
        hparams (omegaconf.DictConfig): OmegaConf 的配置对象（通常是嵌套结构）

    Returns:
        dict: 展平后的字典，键为点分隔字符串，值为基本类型或列表

    Raises:
        RuntimeError: 当遇到不支持的类型时抛出
    """
    def _to_dot_dict(cfg):
        """内部递归函数，处理嵌套结构并生成展平字典"""
        res = {}
        for k, v in cfg.items():
            # 处理空值
            if v is None:
                res[k] = v
            # 处理嵌套的 DictConfig
            elif isinstance(v, omegaconf.DictConfig):
                # 递归调用自身处理子配置，并将键合并为点分隔形式
                # 例如：父键 "model" + 子键 "lr" → "model.lr"
                res.update({k + "." + subk: subv for subk, subv in _to_dot_dict(v).items()})
            # 处理基本类型
            elif isinstance(v, (str, int, float, bool)):
                res[k] = v
            # 处理列表类型
            elif isinstance(v, omegaconf.ListConfig):
                # 解析列表中的变量（如 ${env:VALUE}），并转换为 Python 列表
                res[k] = omegaconf.OmegaConf.to_container(v, resolve=True)
            # 其他类型不支持
            else:
                raise RuntimeError('The type of {} is not supported.'.format(type(v)))
            
        return res
    
    return _to_dot_dict(hparams)

def seeding(seed=0):
    """
    设置深度学习训练中的随机种子和相关库的确定性配置，确保实验的可复现性
    """
    torch.manual_seed(seed)                     # 设置 PyTorch 的 CPU 随机种子
    torch.cuda.manual_seed(seed)                # 设置 PyTorch 所有 GPU 的随机种子
    
    np.random.seed(seed)                        # 设置 NumPy 的随机种子
    random.seed(seed)                           # 设置 Python 内置 `random` 库的随机种子
    
    torch.backends.cudnn.enabled = True         # 启用 cuDNN
    torch.backends.cudnn.benchmark = True       # 允许 cuDNN 自动寻找最优卷积算法
    torch.backends.cudnn.deterministic = True   # 强制使用确定性算法，确保结果可复现



def run_trainer(trainer_cls):
    """
    读取参数文件和传入的命令行参数，启动训练/测试流程
    
    例如：
    python trainer_sup.py run_mode=train run_cfg=exp/log/<<folder_name_with_sup>>/config.yml
    python trainer_sup.py run_mode=test run_ckpt=exp/log/<<folder_name_with_sup>>/ckpt_latest.pth
    
    Args:
        trainer_cls: 一个模型类，不是模型类的具体实例，而是模型类名
        
    Returns:
        None: 解析参数以后，会创建模型实例，并启动模型
        
    Raises:
        RuntimeError: 如果没有正确提供训练类型会报错
    """
    cfg_cli = OmegaConf.from_cli()          # 从命令行读取参数
    assert cfg_cli.run_mode is not None     # 确保设置了训练方式
    
    if cfg_cli.run_mode == "train":         # 训练
        assert cfg_cli.run_cfg is not None  # 确保提供了Config配置文件路径
        cfg = OmegaConf.merge(              # 将yml中的参数与客户端参数汇总
            OmegaConf.load(cfg_cli.run_cfg),
            cfg_cli
        )
        
        OmegaConf.resolve(cfg)
        cfg = omegaconf_to_dotdict(cfg)     # 转换为 Pyhton 字典或类似对象
        seeding(cfg['seed'])                # 设置随机种子
        trainer = trainer_cls(cfg)          # 创建模型实例
        trainer.train()
        trainer.test()
        
    elif cfg_cli.run_mode == "test":                                # 测试
        assert cfg_cli.run_ckpt is not None                         # 检查是否提供了ckpt文件
        log_dir = str(Path(cfg_cli.run_ckpt).parent)                # 获得ckpt所在目录路径
        cfg = OmegaConf.merge(                                      # config.yml文件中的参数和命令行参数合并
            OmegaConf.load(os.path.join(log_dir, 'config.yml')),
            cfg_cli,
        )
        
        OmegaConf.resolve(cfg)                      # 解析参数
        cfg = omegaconf_to_dotdict(cfg)             # 将解析后的参数转为字典
        cfg['test_ckpt'] = cfg_cli.run_ckpt         # 将ckpt路径加入配置
        seeding(cfg['seed'])                        # 设置随机种子
        trainer = trainer_cls(cfg)                  # 创建训练器实例
        trainer.test()                              # 执行测试
    
    else:
        raise RuntimeError(f'Mode {cfg_cli.run_mode} is not supported.')    


if __name__ == '__main__':
    print("hello")


