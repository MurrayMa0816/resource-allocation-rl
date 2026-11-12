第五章：当前代码与正式实验入口
===============

代码均为实际文件。ROOT保独立维护版本，HY完整包保同构实体副本；数据由RESOURCE_ALLOCATION_DATA_ROOT/FINAL_DATA_ROOT指定。
先看代码角色与论文小节.csv。_frozen是完整原始科学源码，_internal/foundation是当前实现依赖，不是额外旧实验模型。

安全入口
------------

python -B -m chapters.chapter05.training.worker
python -B -m chapters.chapter05.evaluation.pipeline
无参数只核285源文件SHA并显示当前100608配置；有--config默认validate-only，只有另给--execute才运行。
method/environment/rollout是薄门面，真实实现仍在_internal；公开worker不再默认使用历史61440。正式285源胶囊保原签名，Full/OnlyJ/OnlyC/ScalarValue各有真实身份。
根requirements-chapter05-torch.txt记录维护依赖；不与第四章JAX环境混锁。

图件与源
------------

plots保精确plot.py、renderer、drawio和小配置，不复制PDF/PNG结果。每图data_package.json明确HY中完整数值输入位置。
正式source_manifest和许可证保持原字节；论文展示CSV、模型、训练/DEV/TEST由同章资料索引查询。
原始队列/launch/register脚本仅科学来源保留，不作为当前公共派发入口。每章在全新Python进程使用其原生命名空间，禁止混入另一章allocation23缓存。
