洪庙支渠灌溉水资源配置推演
================================================

按论文顺序打开下面的实际小节目录。原始模型和运行包在本章原目录中集中保留，
跨章共用模型仍在本数据包内单份保存；所有CSV均给出包内实际路径。

先查论文小节对照表.csv，再查模型用途对照表.csv、图表与文件对照表.csv。

论文小节
--------

6.1 推演环境构建
  chapters/chapter06/paper/6.1_推演环境构建/
6.1.1 中稻灌溉配水任务
  chapters/chapter06/paper/6.1.1_中稻灌溉配水任务/
6.1.2 强化学习任务建模
  chapters/chapter06/paper/6.1.2_强化学习任务建模/
6.1.2.1 状态与观测
  chapters/chapter06/paper/6.1.2.1_状态与观测/
6.1.2.2 动作与约束
  chapters/chapter06/paper/6.1.2.2_动作与约束/
6.1.2.3 状态转移
  chapters/chapter06/paper/6.1.2.3_状态转移/
6.1.2.4 奖励与目标
  chapters/chapter06/paper/6.1.2.4_奖励与目标/
6.1.3 推演环境实现
  chapters/chapter06/paper/6.1.3_推演环境实现/
6.2 配置推演与结果分析
  chapters/chapter06/paper/6.2_配置推演与结果分析/
6.2.1 推演设置与评价指标
  chapters/chapter06/paper/6.2.1_推演设置与评价指标/
6.2.2 配置过程与水分响应
  chapters/chapter06/paper/6.2.2_配置过程与水分响应/
6.2.2.1 全期用水安排
  chapters/chapter06/paper/6.2.2.1_全期用水安排/
6.2.2.2 区域配水过程
  chapters/chapter06/paper/6.2.2.2_区域配水过程/
6.2.2.3 相邻两日的配水调整
  chapters/chapter06/paper/6.2.2.3_相邻两日的配水调整/
6.2.3 配置成效与条件分析
  chapters/chapter06/paper/6.2.3_配置成效与条件分析/
6.2.3.1 学习方法的配置成效比较
  chapters/chapter06/paper/6.2.3.1_学习方法的配置成效比较/
6.2.3.2 区域产量分布
  chapters/chapter06/paper/6.2.3.2_区域产量分布/
6.2.3.3 供水条件与配置成效
  chapters/chapter06/paper/6.2.3.3_供水条件与配置成效/
6.2.3.4 天气变化后的配水调整
  chapters/chapter06/paper/6.2.3.4_天气变化后的配水调整/

原始资料
--------

figures/
paper_tables/
studies/

MODELS.json中的method保留实现家族；paper_method/variant用于区分论文实际方法与变体。
原始协议和源码不因整理改写，历史绝对路径仅记录来源，当前查阅使用本章CSV。
本包用于最终实验核查；论文TeX、整稿PDF和日常维护代码应另行备份。

第六章完整40项评价/2698条轨迹另见评价与轨迹总表.csv；原始输入与协议总表.csv列固定输入。
三张论文表除原文外另有参数/数值核对CSV。正文与补充验证身份分别标明。
