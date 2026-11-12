消融算法与效果
==============

variants.py 是当前活动消融入口，与worker.py实际调用同一文件。
full：完整Full；only_j：只用原任务指标J；only_c：只用达标比例C；
scalar_value：将C/J混合为单个标量价值监督。具体行为以本文件为准。
Full调用共享Full学习器，三消融保留共同物理环境、策略初始化及采样流程。
本轮不修改消融定义或重新训练。

REGISTRY.json登记上述组别；对应已完成效果从effects/进入，实际指向results/02_ablations。
实验计划和历史协议见experiments/02_ablations/，不得将登记说明当成自动启动配置。
_internal中的旧cross_server/variants.py仅历史兼容，不是当前修改点。
