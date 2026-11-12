本章算法Full
============

models.py：策略与价值网络构造；values.py：终点达标比例C、原任务指标J的标签及优势；
learner.py：Full学习更新；common.py：方法共用工具。
initialization.py：本章私有的空间、当期总量双输出头随机初始化及模型身份检查。
这里是日常查看和修改本章算法的入口。每个文件链接同一份_internal源码，不复制第二套。
不要直接以python method/learner.py启动；统一通过总工程 ./project chapter5 运行，保证包导入和环境版本一致。
原Full与新P1/P2路线的开发配对结果在results/05_method_redesign；它们不是新采用的外部对比方法。

当前初始化
----------

新建Full及同策略骨架的OnlyJ、OnlyC、ScalarValue均采用
chapter5-both-heads-default-random-v1。空间输出层与总量末层的权重、偏置
保留原生Linear默认随机值：先按原构造种子以float32抽样，再转float64。
空间种子为31000+训练种子，总量种子为83100+训练种子。
相同种子可复现，不同种子的初始网络和状态驱动决策会变化。

冻结原生依赖不修改。本章工厂重放原构造随机流，仅恢复原来被清零覆盖的
四个输出参数张量；其他网络、价值函数、探索尺度、字典及物理解码保持。
周期余额参考及区域配置可行投影保持，末期全额分配等物理约束仍有效。
随机初始化不保证初始指标较低，也不保证学习曲线单调；同质对象或约束
唯一确定投入时仍可能有相同的物理配置。

全额Full训练统一通过launcher train使用。worker的protocol.initialization
登记profile及两头的种子和初始参数哈希，终点重载先核初始化身份。
旧模型、旧曲线和冻结读取代码保留为历史，不能重标为新初始化训练结果。
