第三章独立开发目录
==================

本章唯一活动五种对比方法为C-PPO、PFO、Gradient PPO、Reflective PPO、PPO+。PFO已由用户正式选定替换CrossQ，选定不再取决于是否接近CrossQ分数。CrossQ源码、配置和结果保留为历史兼容，不作为第六个活动方法；旧冻结诊断中的未合格结论保持原样。

学习实现只在src/chapter3_candidates维护，allocation23/chapter3_candidates为同一实体兼容入口。pfo.py使用已验收的PPO特征稳定正则，spo.py仅提供采集/GAE/保存恢复基础，不注册为外部方法。论文与MIT许可、任务结构适配见同目录pfo_provenance.json及licenses/PFO_MIT.txt。

从总工程查看::

    ./project chapter3 inspect
    ./project chapter3 baseline inspect --method pfo
    ./project chapter3 ablations

A、四个总量消融和原四个外部对照的100608步五种子正式批次已完成，原130任务身份及结果不变。PFO正式双环境五种子共10项尚未开始；旧两种子24576或98304步诊断不能冒充已完成。当前PFO精确源仍要求512整批，100608=196*512+256，构造/尾批接口必须适配并通过真实工程门禁后才能安排正式10项。当前CLI在构造物理环境前直接拒绝该预算，并明确提示尾256待适配；完整512整批的已有预算仍须显式令config.total_steps与--steps一致，不会先跑到旧时钟上限再报错。

本轮只更新代码和正式名册，不启动正式训练、不改正在运行的longscope/coordinator、冻结实验或论文。每次运行仍需新的实体HY输出，完整数值曲线和一个固定终点模型，无磁盘回放/Adam/周期模型。历史CrossQ端点仍应按原冻结source校验评价，不放宽来源门禁。

正式用户决定：releases/pfo_user_adoption_20261106.json。当前五法指针：/hy-tmp/resource_allocation_results/chapter3/LATEST_FORMAL_METHODS.json。
