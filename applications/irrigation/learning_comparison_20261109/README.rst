第六章当前三法学习比较维护代码
========================================

当前采用SeDVal、PASTA和COLA各3个400万固定3200回合端点。比较使用共同26×32 DEV；没有执行三法共同独立TEST。CAPQL已退役，不在当前模型清单。完整模型和训练/DEV原始记录在DATA本章studies。

python launcher.py compare inspect仅读取当前9模型身份与存在性；launcher.py inspect/check/redraw走统一只读章入口，不派发训练、工程、历史controller或新TEST。current_comparison.py为只读元数据入口。

paths.py通过公共portable_paths及私有c6_paths读取实体DATA与代码根；原C6环境从ROOT/frozen/chapter06/reasonable89_20261109取，PASTA/COLA共享ROOT/chapters/chapter05/_frozen/formal285完整原285源。维护适配器只改文件定位，原生学习函数、配置与数学不变。原实际执行胶囊在DATA/studies的source/formal_attempt0内保持原字节。

旧cola_launcher.py、execution/prepare*、worker与analysis是原执行来源，保原来源说明，不经当前入口调用。当前论文复绘直接使用DATA本章figures完整包；旧analysis/paper_learning_comparison.py不作为当前图入口。原协议绝对位置只作历史来源，不要求恢复旧ROOT或HY目录。
