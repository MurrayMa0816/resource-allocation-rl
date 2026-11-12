第四章：近期完整轨迹复用的前缀加权对照
====================================

结果：/hy-tmp/resource_allocation_results/chapter4/reuse_optimization_20261102
代码：allocation_reuse。环境、A决策生成、原F1均固定且由allocation_efficiency导入。
F1为原始对照；P1原始冻结排他前缀权重；P2前缀截断后按每时刻旧4归一。
三组同CPU推理镜像和CUDA学习器，保留CUDA原始高斯噪声流。
先pytest、独立审查、6工程任务，通过后18正式任务全部并发。
主指标固定质量确认所需新训练物理步Nq，辅以AUC、独立质量、完整成本。
具体方案、限制与冻结数值见结果目录protocol.json和方案与判定规则.rst。
