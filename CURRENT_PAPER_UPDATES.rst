当前论文新增两表的资料入口
================================

本说明补充表6.4、表6.5的查阅入口。原实验包的83小节与54图表导航形成较早，
未自动包含这两个新增表项；按原lookup查表6.4或6.5可能无结果。可直接使用
下列现成记录查阅，不需要重新训练、载入模型或执行新的测试。

本次只读取保存的NPZ数值数组、metadata与模型索引，并核对显示值；阶段水量
仅将已保存的逐日分配量相加，终点指标读取已有记录。没有启动环境、策略计算、
推理或新实验，原包和原科学映射未改。

共同路径与实例身份
------------------

下文的共同路径前缀为：

.. code-block:: text

   chapters/chapter06/studies/reasonable_allocation89_20261109/

各条记录的文件路径均相对源码与资料合并后的发布根目录。
已登记的基准实例见此前缀下 ``formal/evaluation/representative_case.json``：
训练种子6400、天气ID125、天气种子3900000125。该记录保存选择依据，
不能将125号单例当作前32条天气、三次训练的96组配对汇总。

表6.4：三档供水方案
------------------

题名：同一天气下三档灌溉供水方案的阶段用水与终点结果对比表。
论文标签：``tab:irrigation89-supply-plan-process``；所在小节6.2.3.3。

表中三列方案均取128条天气记录中 ``weather_id == 125`` 的一条；在这些文件
中其零起始行号为125。每个目录均有 ``results.npz`` 和 ``metadata.json``：

.. list-table:: 相同种子与天气下的三档方案
   :header-rows: 1
   :widths: 16 54 30

   * - 全期水量
     - 共同前缀下的记录目录
     - 模型索引身份
   * - 350万立方米
     - ``formal/evaluation/trained_budget_3500000__seed6400/attempt0/``
     - ``SeDVal_供水350万_seed6400``
   * - 400万立方米
     - ``formal/evaluation/main__seed6400/attempt0/``
     - ``SeDVal_供水400万_seed6400``
   * - 450万立方米
     - ``formal/evaluation/trained_budget_4500000__seed6400/attempt0/``
     - ``SeDVal_供水450万_seed6400``

350万与450万记录为各自供水条件下训练所得策略，不是把400万模型换供水量后
直接执行的 ``fixed_budget_*`` 补充条件记录。模型实体及SHA见根目录
``chapters/chapter06/MODELS.json``；上述metadata中的 ``model_sha256`` 与登记
模型逐一一致。

从 ``q_m3`` 读取每日向全部区域分配的水量，从 ``balance_m3`` 读取分配后
剩余量，日期见 ``date``。各阶段的零起始日范围为返青0—8、分蘖9—45、
孕穗46—61、抽穗62—71、乳熟72—80。原 ``calendar_stage_index`` 将分蘖分为
湿润段与晒田段，表6.4将这两段合写为分蘖期；应先累加未舍入的每日数值，
再换算为万立方米并保留两位小数，不能相加已舍入的分段显示值。

终点达标区域数、达标率和按面积加权的平均相对产量分别读取
``coverage_count``、``C``、``JA``。单例的三档结果依次为
61/89与68.54%、81.97%；89/89与100.00%、92.14%；
89/89与100.00%、97.66%。这些单例值与图6.7的跨天气、跨种子汇总作用不同。

表6.5：天气变化后的终点结果
----------------------------

题名：所选天气实例及参考蒸散率变化后的灌溉用水分配方案终点结果对比表。
论文标签：``tab:irrigation89-weather-plan-outcomes``；所在小节6.2.3.4。

“未调整”一行取上述 ``main__seed6400`` 的天气ID125；其余四行取下列
单例专用目录的 ``results.npz`` 与同目录 ``metadata.json``。这些文件只有
一个天气实例，其数组零起始行号是0，不是125；须同时核对 ``weather_id``：

.. list-table:: 四条已保存的单例过程
   :header-rows: 1
   :widths: 13 21 48 18

   * - 条件
     - 分配方式
     - 共同前缀下的目录
     - 达标数/平均产量
   * - 降低15%
     - 执行此前方案
     - ``formal/evaluation/descriptive_feedback_low_replay__seed6400__weather125/attempt0/``
     - 89/89；92.40%
   * - 降低15%
     - 根据新观测重新确定
     - ``formal/evaluation/descriptive_feedback_low_feedback__seed6400__weather125/attempt0/``
     - 89/89；92.42%
   * - 提高15%
     - 执行此前方案
     - ``formal/evaluation/descriptive_feedback_high_replay__seed6400__weather125/attempt0/``
     - 87/89；91.85%
   * - 提高15%
     - 根据新观测重新确定
     - ``formal/evaluation/descriptive_feedback_high_feedback__seed6400__weather125/attempt0/``
     - 86/89；91.81%

四条记录都对应 ``SeDVal_供水400万_seed6400``，metadata中的模型SHA与
基准模型相同；只改变指定日期的参考蒸散率条件及执行方式。记录中
``forcing_transform`` 明确7月31日至8月2日（零起始日62—64）的乘数0.85或1.15。
对应配置还保存在共同前缀下 ``formal/configs/descriptive_feedback/``。

此四条专用记录依预先保存的 ``protocol/representative_feedback_contract.json``
补存天气ID125的描述过程，不纳入每类96组配对汇总。原统计用的
``feedback_low_*`` / ``feedback_high_*`` 与这里的 ``descriptive_feedback_*``
各有身份，不能互换。

保全与覆盖范围
--------------

两表所需的上述七份结果NPZ、同目录元数据、实例选择合同与三份模型均已在
原完整实验包中找到，未发现为这两表必须另行补跑的输入。原包没有单独以新表号
命名的成品CSV或TeX表快照；本文件补充其查阅路径和单例身份，不伪称原导航已
更新到56图表。当前论文的节号、题名和页码对照另见 ``PAPER_MAP.rst``。
