第六章当前八图源码
========================

各目录为当前完整图包中的绘图源码、固定几何、CSV/数组及小配置的实体副本；逐文件SHA与DATA包一致，未修改renderer。PDF/PNG/SVG出版输出及临时复绘缓存不复制到代码交付。

各图DATA_PACKAGE.json标明其完整数据图包；用公共portable_paths.chapter_data_root(6)/figures/目录名定位。当前章入口redraw在完整DATA图包运行保存记录复绘，输出到单独目录；环境交互/模型推理不参与。图6.2为Drawio图源，其source.drawio在本目录；七个Python入口在其余图包。

ROOT维护图源改动应与对应DATA本地图包显式同步并重新登记SHA，不用软/硬链接拼接。原始图输入不改变。
