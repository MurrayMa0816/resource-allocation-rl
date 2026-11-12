PASTA：已有修正版和历史最高配方复核候选
=====================================

本目录是待复测候选，不代表正式采用。来源为作者RSS2026仓库，固定commit
745d769740b6ce648c7ea8209933733dea8a1eab；作者根目录未声明代码LICENSE，
不能冒称MIT或把论文许可当代码许可。_native.py原字节保留。

入口 Agent(obs_dim,action_dim,seed=0,device='cpu',config=None)。
configs.json为configurations[domain]['current_recipe']。
沿用既有GAE done[t]、PCGrad objective-ID两项修正；本轮没有再改学习算法。
radio沿用R2 lr1e-4，v5沿用R1 lr3e-4，其余是原已声明参数。

on_policy=True，action_mode=raw。每个完整回合panel调用train_preferences，
每tick sample_info返回actions/logp/values；整panel后调用update_rollout。
trajectory中obs[T,B,O]、actions[T,B,A]、logp[T,B,1]、values/rewards[T,B,2]、
dones[T,B,1]。所有panel为真实完整回合，只有最后tick的done=1与终奖非零。
B=8、每8个panel优化一次；radio T32、v5 T24。因此首次真实优化分别在
2048、1536训练步，起点后短暂平线是尚未优化，不补点、不移动横坐标。
61440/24576两种预算均落在完整优化边界；不能把不同预算混成主比较。

DEV可在精确512步中途执行，但必须保留未完panel和全部训练状态/RNG。
state_dict为RAM-only完整恢复（含Adam），固定终点请用checkpoint_state_dict，
它只含actor及必要公共观察统计，不含critic/Adam/回放。
manages_observation_normalization=True；选定配方内部归一化关闭，不叠加外部映射。

历史应急两seed明显学习；无线9731仅略过预登记净增门槛，末段有平台/轻微下降，
而且低于中性规则，不能称其持续强收敛。新数据之前不改变这个限制。
本轮仅CPU合成/零物理工程核查；不因此宣布真实CUDA或正式实验验收通过。
