第六章：灌溉配置推演
========================

本章保留89区、81日的环境、SeDVal方法和PASTA/COLA对比接口。按下列浅目录查代码：environment/是环境与固定输入清单，method/是三法与原执行源，training/是训练和DEV记录身份，evaluation/是原TEST及当前图入口。完整Python包保持原结构。

日常入口为python chapters/chapter06/launcher.py inspect、check、compare inspect或redraw；path打印本章实体数据目录。仅检查与保存记录复绘，不载入模型、不运行环境或重启训练。DATA位置由公共portable_paths按显式环境或本地INDEX识别。

维护应用在applications/irrigation；原SeDVal科学140源在frozen/chapter06/reasonable89_20261109，C5共享原285源在chapters/chapter05/_frozen/formal285。冻结源保持原SHA；外法维护适配只改变文件定位。历史launcher、准备器与controller属于原执行来源，不是本章日常入口。

最终数据保留15端点（9个SeDVal，PASTA/COLA各3），40项评价、2698条完整轨迹。外法比较只用共同DEV；SeDVal独立TEST、初始化重建、各供水重训和固定策略补充分别登记。当前图从本章数据figures完整包复绘。
