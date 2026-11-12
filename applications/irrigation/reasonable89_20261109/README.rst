第六章89区当前维护应用
============================

当前完成9个固定3200回合SeDVal模型（三供水×三seed）及40项正式评价，共2698条完整81日推演。原科学source/、method/及execution/common.py和worker_train.py共140源保持既有release7143fa7a029532bc0ca82275652f595853c90ab35992bdc6bd3ff6704aa0f487，ROOT/frozen/chapter06/reasonable89_20261109保存原执行胶囊。

只读维护入口：python launcher.py inspect、check、compare inspect或redraw；统一章入口在chapters/chapter06。数据位置由公共portable_paths定位实体DATA，当前记录和图件均在DATA本章目录。检查不导入环境或加载模型，复绘只用已保存记录。

评价evaluation/保留原执行的可审读镜像和原manifest；140科学源与原评价封印不作路径热改。历史worker/准备器不作日常入口，现入口不运行或恢复旧训练/TEST。维护包装与不可变科学快照分开登记。visualization/保留原图源身份，当前论文请用DATA本章figures完整包的plot.py；不依赖旧ROOT/HY展示路径或任何文件系统链接。

供水350/450万正文采用各档重新训练模型；固定400万策略换供水为补充。外法只用共同DEV；SeDVal另有独立TEST。初始化TEST按seed重建episode0网络，不是另有3个初始模型文件。
