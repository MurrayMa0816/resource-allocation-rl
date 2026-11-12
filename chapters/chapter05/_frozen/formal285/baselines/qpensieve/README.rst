qpensieve 候选复核实现
================================

入口：from baselines.qpensieve import Agent, load_config。
当前仅供候选复核，未加入正式外部对比名单，不恢复旧队列。
代码实体位于HY，不导入旧日期包的训练器或环境。统一worker必须使用当前全额Env。

配置：configs.json，按profiles→legacy_recipe/fixed→radio/v5读取。
Agent自己管理train-only观察归一化，worker调用observe_training一次/真实步，勿再包第二层。
自有RNG用rng_state_dict保存；CAPQL使用全局Torch抽样，worker仍须隔离全局Python/NumPy/Torch/CUDA状态。

QP fixed仅修复作者Xavier/零bias初始化；原初始化与修复初始化不同，必须保留各自真实pi0，不能声称同起点数值配对。

历史五轮优化与真实增长、无线仍弱于同情景规则的边界见provenance.json和historical_review.json。
CPU合成检查不等于物理训练已改善；真实GPU门禁与学习趋势由独立新scope验证。
旧配方及源保留，不以新结果覆盖负结果。
