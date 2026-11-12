数据、文献与素材来源
====================

本文件提供资料来源导航，并区分第三方原始资料、项目转换输入和研究设定。
它不统一授予所有数据、模型、文献或字体的复用许可。实际分发范围以本发布版本
的 ``release_manifest.json`` 和 ``LICENSE_SCOPE.rst`` 为准。

EuroSAT RGB 图像输入
--------------------

EuroSAT 由 Patrick Helber、Benjamin Bischke、Andreas Dengel 和 Damian Borth
等研究者发布，基于 Copernicus Sentinel-2 遥感图像。本项目使用其 RGB 版本作为
图像传输任务的输入，不将其表述为本项目自采数据。

作者官方项目 README 明确将数据集列为 MIT 许可，并提示使用者考虑 Copernicus
Sentinel 数据条款。官方 LICENSE 的版权信息为 ``Copyright (c) 2023 Patrick
Helber``。使用或再分发时应保留相应版权、许可及归属声明，不把本项目第一方
许可状态覆盖到 EuroSAT 数据上。

* `EuroSAT 作者官方项目与引用要求 <https://github.com/phelber/EuroSAT>`_
* `EuroSAT 官方 MIT 许可证 <https://github.com/phelber/EuroSAT/blob/master/LICENSE>`_
* `作者发布的 Zenodo 数据记录 <https://zenodo.org/records/7711810>`_
* `项目所指 Copernicus Sentinel 数据条款 <https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice>`_

官方许可证全文随本发布版保存在 ``third_party/EuroSAT/LICENSE``；内容从作者项目
提交 ``b05b5e5b87d150a802237dd4a6d2d1907631343b`` 的 LICENSE 原样取得：
`固定版本许可证来源 <https://github.com/phelber/EuroSAT/blob/b05b5e5b87d150a802237dd4a6d2d1907631343b/LICENSE>`_。
该随附文件为 1071 字节，SHA256 为
``73b8c5d100ff46a84845702b9e78d76ddb1effaacdb386b91dc07061c6123362``。

本包 ``datasets/eurosat_rgb/images.npy`` 保存已经解码的 RGB uint8 像素，形状为
``(27000, 64, 64, 3)``。输入来源、原压缩包 MD5、预处理、划分和样本对应见：

* ``shared/data/EuroSAT_RGB/manifest.json``。
* ``shared/data/EuroSAT_RGB/split_manifest.json``。
* ``shared/data/EuroSAT_RGB/image_records.json``。

现有记录中的源压缩包为 ``EuroSAT_RGB.zip``，MD5 为
``f46e308c4d50d4bf32fedad2d3d62f3b``，与上述官方 Zenodo 记录公布的值一致。
记录的转换为 Pillow 将原 JPEG 解码到 RGB uint8，保留原 64×64 尺寸，不做缩放、
裁剪、增强或再次 JPEG 压缩；参考对象是 JPEG 解码像素。

本项目使用自定义分层 70/15/15 划分，分别为 18900、4050、4050 个样本，
划分种子见 ``split_manifest.json``。这些划分是本项目研究约定，不能称为作者
官方训练、验证、测试划分。本说明核对既有来源记录和数组头，没有重新下载、
重新预处理或重新执行图像实验。

第六章区域、天气与文献推算
--------------------------

第六章当前推演使用 89 区输入；历史来源文件中的 ``hongmiao93`` 名称继续用于
标识原始资料与执行来源，不能据此把当前推演单元数改为 93。入口如下：

* ``datasets/chapter06/regions_89.csv``：当前区域输入。
* ``datasets/chapter06/weather_originals/hongmiao_supply_weather_alcar581/weather_manifest.json``：
  天气来源、经纬度、请求 URL、字段单位、时间标准和文件身份。
* ``chapters/chapter06/studies/reasonable_allocation89_20261109/formal/fixed_inputs89.json``：
  当前实验固定输入。
* ``chapters/chapter06/原始输入与协议总表.csv``：按章查找固定输入与原始协议。
* ``applications/irrigation/THIRD_PARTY_NOTICES.rst``：第三方来源及适用范围说明。
* ``applications/irrigation/reasonable89_20261109/source/_frozen/irrigation_ch6/_core/data/hongmiao93/``
  中的 ``source_registry.csv``、``case_model.json`` 与 ``manifest.json``：文献、
  参数、模型假设与输入来源；同名资料也可能保存在对应冻结执行源中。

天气来自 NASA POWER 项目的区域参考点数据；现有 manifest 明确该点是邻近区域
代理，不是逐区气象站实测，也不是灌区各口门的实际坐标。其 LST 表示地方太阳时。
参考蒸散、条件预报与其他派生量是项目计算，应与 NASA 原始字段分别理解。
NASA POWER 的来源与使用信息以其官方资料为准：
`NASA POWER 官方 API 文档 <https://power.larc.nasa.gov/docs/services/api/temporal/daily/>`_。

``source_registry.csv`` 列出公开面积表、农业文献、FAO56、气象数据和初始水深
指导等来源，并逐项注明适用范围。数值转录、邻区参数迁移、冻结随机赋值、
初始状态与供水控制属于不同来源或研究设定；不能统称为灌区现场观测或逐区标定。
已保存的模型输入和结果不因此自动成为可任意复用的第三方原始资料。

第三方全文与字体
----------------

对外资料不包含第三方论文、专利或技术资料的全文副本及其全文提取文本、网页
抓取件；读者应通过来源登记中的链接自行获取。保留引用、数值出处和项目推算
说明，不将可在线阅读等同于获得整份文献再分发授权。

第三方代码和允许随附的字体保留原有许可证及归属文件，位置见各自目录与
``LICENSE_SCOPE.rst``。已附许可的 Droid 字体按其原声明保留；分发依据未明确的
Windows 字体与 ``assets/fonts/figure/ThesisFigureText-Regular.otf`` 不随发布版
提供。未提供的字体可能需要使用者自行取得相应授权或选择替代字体，复绘字形和
布局可能随之变化。

本文只核对了所列来源与已有记录，不宣称已为所有第三方资料取得额外的许可。
