第四章：修正基础学习器后验证快照控制变量
======================================

代码：/root/gamewar/chapter4_sample_efficiency_v5
结果：/hy-tmp/resource_allocation_results/chapter4/calibrated_snapshot_20261101

本轮只修正V4的零baseline与一次微小Adam步：改为采前冻结的状态价值baseline、
按经验KL尺度的单次自然梯度更新。分配结构仍是第三章A；环境与终奖均不改。
当前critic用已完成新回合的MC终奖训练，只用于下个参考周期的baseline。

B1/F1更新源码保持不变。P2/S2共享采样、critic训练、baseline与NPG后端，
仅S2把参考梯度和精确跨策略差分用于更新。保留有限参考误差，禁止声称条件无偏。
Adam只用于共同32回合热身；postwarm手动NPG更新、CG/FVP与回退单独计数。

先合成CUDA验证及独立数学审核，再4组96回合工程验证。
之后一次固定6组240回合pilot检查P2是否持续学习。每域要求：
mean(dev192,dev240)-dev48 >= max(1, .5*max(B1同段growth,0))，且192/240均高于48。
任一域失败便停止，不调参、不重复pilot、不启动S2正式训练。

仅当门槛通过，才运行24组=4版本×2域×3新种子90—92×480回合正式比较。
主指标是同质量目标确认所需新交互；包括参考刷新，未达标保留删失。
pilot不并入正式效应。所有结果与成本保存在/hy-tmp，不修改旧V1—V4。
