第四章当前全额分配验证代码
========================

当前实现位于本目录，工作区目录是HY源码的链接，不复制或改写历史原件。
共同环境代码为 env/chapter4_full_spend_env/，A/F1入口为reference_worker.py，
PROPS适配入口为communication/run_props_causal.py，ADDQ配对入口为
native_diagnostic/addq_gamma_worker.py。

本版本使用第三章2026年11月4日的公开[L,U]全额分配规则：每回合全部轮次的
实际资源之和等于初始总预算，各轮仍可动态选择不同总量，不突破对象上限。
不增加即时奖励或未来信息。具体来源和已通过的实际测试见env/acceptance/。

所有32项960回合开发测试已完成，固定配置、原始数据、单终点模型与报告位于：
/hy-tmp/resource_allocation_results/chapter4/full_spend_communication_repair_20261104/

原chapter4_recent_baselines及2026年11月2日冻结树保留历史复现，仍可能引用
旧的允许余量环境；它们不代表这个新全额版本。原暂停队列没有恢复。
本轮已完成任务不应重启或覆写，新训练必须登记新输出目录和配置。

独立验收与重新绘图（只读原结果，不训练）::

  python independent_acceptance.py --require-complete
  python verify_public_replays_numpy.py
  python build_report.py --acceptance ../results/independent_acceptance/latest.json

若从工作区链接位置运行，最后命令应使用上面的HY结果根给acceptance绝对路径。
报告和全部验证仅支持两个训练重复的开发结论，不等同五种子独立终测。
