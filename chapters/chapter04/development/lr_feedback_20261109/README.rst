第四章学习率反馈四臂开发测试
================================

本目录仅用于新注册开发实验，原正式代码、模型和论文结果保持。
worker.py由原double_random_formal的f1_worker.py窄派生，完整轨迹采集、
确定性DEV、终点权重重载、物理守卫与全部压缩原始输出沿用父逻辑。
不提供TEST入口，不加载旧模型，不存优化器、回放或周期模型。

四臂为original_feedback、fixed_3e4、fixed_1e4、relaxed_feedback。
fixed_1e4从预热开始使用1e-4，其余均从3e-4开始。
宽松反馈仅把批后控制参考区间改为[.05,.10]，保持1.03倍调整、
原[1e-5,1e-3]范围与0.2/1.5广义裁剪。所有SGD、回溯、轨迹替换
仍由原FrequentPPO执行。批后覆写仅影响下一批学习率。

注册CLI要求--registry、--case-id和--expected-registry-sha。
注册jobs中method/domain/seed/steps/output/device/train_master/dev_master/
eval_every/dev_cases/engineering_gate与CLI逐项一致。
开发固定49152步，工程固定1536步；DEV固定每768步32情景，含0与终点。

Python运行时为resource_allocation_rl/environments/chapter04/runtime/torch/bin/python。
PYTHONPATH须包含resource_allocation_rl/frozen/dependencies/chapter04/
recent_baselines_formal_20261102_source。零环境检查入口zero_env_check.py。
工程与正式开发均由独立scope注册和root调度，此目录本身不自动启动任务。
