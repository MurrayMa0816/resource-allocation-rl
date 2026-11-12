第四章：有限终奖监督下的信用学习
==============================

本包独立于第二、三章代码，固定依赖 /root/gamewar/chapter23_final。
环境通过Task(domain,3,'A')建立：通信100终端32轮，应急120区域24期。
环境、A配置结构、精简状态与原生终点奖励保持当前版本。
算法在回合结束后使用完整公开轨迹训练，不向执行actor提供未来或隐藏模型。

实现的候选是基于原论文的适配，并非原论文逐参数复现：

* rudder：因果GRU回报分解、末端补偿、双价值及GAE，配套gae对照。
* cocoa：joint/product对比贡献模型、独立候选动作积分，配套mc_pg。
* lax：以完整回合策略梯度二阶矩训练控制变量，配套cv_mse与mc_pg。
* cca：未来公开外生信号条件基线，配套同容量cca_state与mc_pg。

mc_ppo忠实移植发布版MC-PPO到GPU，已核对CPU参数更新逐位一致。
PG六组使用单次全批Adam，lr=.003，访问状态完整Gaussian KL上限=.01；
PPO/GAE/RUDDER使用发布版多minibatch设置，lr=.0003。
共同环境交互预算不等于相同计算量，必须报告更新数和耗时。

LAX/COCOA/CCA主机制比较必须针对匹配的mc_pg；与mc_ppo比较判断整体实用收益。
LAX的原生梯度补偿、COCOA的精确理论条件不自动延续到有限学习模型、Adam或KL选步。
CCA的future输入仅来自已经结束轨迹的公开外生字段；actor始终仅有当前状态。
模型来自前批，当前更新后才拟合本批数据，避免同批终奖记忆污染信用信号。

运行示例（输出目录须不存在）：

  python run.py --domain radio --method lax --seed 0 --episodes 480 \
      --eval-every 96 --dev-paths 16 --holdout-paths 64 --device cuda:0 \
      --output /hy-tmp/resource_allocation_results/chapter4/example_fresh_lax

执行 python -m pytest -q 运行解析梯度、数据边界、环境回放及GPU集成测试。
检查点记录完整模型和辅助状态；本轮runner尚无断点续训入口，全部正式训练从零开始。
源代码、环境配置、数据标识和随机数命名空间记录在各任务protocol.json。

目录
----

allocation4/common.py          公共张量与来源记录
allocation4/rollout.py         八个独立环境的批量策略推理
allocation4/methods/           九个候选/对照注册名称
allocation4/training.py        GPU从零训练、评价与结果记录
configs/screen.json            运行前固定的首轮配置与比较
tests/                        正式训练前正确性检查

结果统一保存在 /hy-tmp/resource_allocation_results/chapter4/。
Shapley、需额外示范/反事实模拟/未验证后端的框架见文献审查，本轮不冒充普通终奖方法。
