COLA：历史配方复核与一步OADM标签修复候选
========================================

本目录是待重新测试的候选，未加入正式采用名单。旧源和旧实验不改。

入口 Agent(obs_dim, action_dim, seed=0, device='cpu', config=None)。
从configs.json的configurations[domain][profile]选择完整参数。
profile为legacy或fixed_oadm；只有oadm_one_step_target科学开关不同。
旧选择为radio R4、v5 R5，均n_step8；不同域参数原样声明，不重新选seed/峰值。

Agent为off-policy，action_mode=tanh_latent；共享worker调用act与update。
训练开始每个完整并行回合调用train_preferences(step,total,rng)，返回同一旧法的
连续内点偏好，而非固定.8/.2。评估仍用统一部署偏好.8/.2。

fixed_oadm需要batch额外提供oadm_next_obs：原始同一条转移的真实一步后继观察。
Bellman next_obs仍保留n步后继；累计原奖励和done不改。单动作动力学OADM
不应把t+n后继当成t+1标签。n_step1两profile完整更新逐位等价门禁另存。
没有实际学习结果前，不声称该修复必然提升分数或解释全部退化。

观察归一化由根_support.normalization负责，manages_observation_normalization=False。
radio沿用前2048真实随机训练观察拟合、之后冻结的映射；DEV不更新统计。
共享映射必须同时变换obs、next_obs、oadm_next_obs，避免两套坐标。

state_dict不包含Adam/replay，但不等于完整RAM续训状态。DEV必须另保留
rng、action_generator、learn_generator以及公共collector随机状态。
只保存一个固定终点；不能把开发峰值替代固定终点。

源码来源：COLA NeurIPS2025，https://github.com/yeshenpy/COLA ，
pinned c437df0ef28169e9356ea54505350ffde13d925a。MIT许可保留。
sac_base.py是原有Q-Pensieve来源SAC原语，所需MIT许可亦保留。
细节与历史数值见parallel_latest_20261105/audit/cola_pasta_historical_and_fix_review.json。
