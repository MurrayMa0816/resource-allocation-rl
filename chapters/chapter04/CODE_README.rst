第四章：当前代码与正式实验入口
===============

代码均为实际文件。ROOT保独立维护版本，HY完整包保同构实体副本；数据由RESOURCE_ALLOCATION_DATA_ROOT/FINAL_DATA_ROOT指定。
先看代码角色与论文小节.csv。_frozen是完整原始科学源码，_internal/foundation是当前实现依赖，不是额外旧实验模型。

安全入口
------------

python -B -m chapters.chapter04.run
以上仅metadata；训练/评价还需--method、显式100608新配置和--execute，拒绝重放已有科学输出。
当前外法只有PROPS、SAC+ReDo、MR.Q、BRO(Fast)、MAD-TD；旧FoG/ADDQ/TD-MPC2/PGR不在公开派发列表。
正式fixed_actor_lr五种子与49152步LR开发诊断不得混用。Torch/JAX分别使用根requirements-chapter04-torch.txt和requirements-chapter04-jax.txt，不合并NumPy约束。

图件与源
------------

plots保精确plot.py、renderer、drawio和小配置，不复制PDF/PNG结果。每图data_package.json明确HY中完整数值输入位置。
正式source_manifest和许可证保持原字节；论文展示CSV、模型、训练/DEV/TEST由同章资料索引查询。
原始队列/launch/register脚本仅科学来源保留，不作为当前公共派发入口。每章在全新Python进程使用其原生命名空间，禁止混入另一章allocation23缓存。
