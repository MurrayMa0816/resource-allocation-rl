SeDVal 轨迹采集与消融组件
========================

本目录包含第五章使用的完整回合采集和机制消融组件，由章级运行程序加载。

模块
----

``proposed_rollout.py``
  ``rollout`` 采集完整回合，保存原动作、解码动作、Gaussian 参数、行为对数概率、
  双价值及资源配置结果。``step_callback`` 接收每个真实交互步，
  ``after_vector_step`` 在完整并行 tick 后调用，可用于中途评价而不切断采集轨迹。
  ``result_row`` 整理终点评价、资源使用量、情景计划与轨迹索引。

``variants.py``
  ``Learner(..., variant=...)`` 提供 ``full``、``only_j``、``only_c`` 和 ``scalar_value``。
  ``full`` 使用原生双价值更新；``only_j`` 去除 C 目标和 C 价值监督，``only_c`` 对称处理；
  无效价值分支不参与梯度裁剪。``scalar_value`` 保留双塔容量与初始化，使用
  ``0.8*C + 0.2*J`` 的单一聚合价值监督。消融组的策略输入偏好保持为 0.8，
  动作执行、采集和复用机制保持一致。

固定偏好下的标量递推与双价值递推满足线性对应关系；该控制用于区分价值监督方式，
不作为新的优势估计恒等式。

运行与结果
----------

在完整资料根目录查看第五章入口::

    python3 chapters/chapter05/scripts/run.py --help

实际实验配置见 ``chapters/chapter05/configs/recorded/``，
论文采用的模型与结果分别见该章 ``models/index.json`` 和 ``results/index.json``。
工程接口或短轨迹一致性检查不代替完整训练及终点评价。
