from pathlib import Path
import xml.etree.ElementTree as ET
import json,hashlib
O=Path(__file__).resolve().parent
parent=O

mx=ET.Element('mxfile',{'host':'app.diagrams.net','version':'24.7.17','type':'device'})
diag=ET.SubElement(mx,'diagram',{'id':'ch6-rl-environment-compact','name':'灌溉环境交互流程'})
graph=ET.SubElement(diag,'mxGraphModel',{'dx':'524','dy':'245','grid':'1','gridSize':'10','guides':'1','tooltips':'1','connect':'1','arrows':'1','fold':'1','page':'1','pageScale':'1','pageWidth':'524','pageHeight':'245','math':'0','shadow':'0','background':'#ffffff'})
root=ET.SubElement(graph,'root');ET.SubElement(root,'mxCell',{'id':'0'});ET.SubElement(root,'mxCell',{'id':'1','parent':'0'})
style='rounded=1;arcSize=4;whiteSpace=wrap;html=0;strokeColor=#000000;strokeWidth=1.0;fontFamily=Times New Roman,SimSun;fontSize=14;align=center;verticalAlign=middle;spacing=2;fontColor=#263445;absoluteArcSize=1;'
def node(k,label,x,y,w,h,fill,diamond=False):
 st=style+f'fillColor={fill};'
 if diamond:st=st.replace('rounded=1;','rounded=0;')+'shape=rhombus;'
 c=ET.SubElement(root,'mxCell',{'id':k,'value':label,'style':st,'vertex':'1','parent':'1'})
 ET.SubElement(c,'mxGeometry',{'as':'geometry','x':str(x),'y':str(y),'width':str(w),'height':str(h)})
node('initialize','回合初始化',8,5,112,32,'#F1D9B5')
node('observe','日初观测',8,61,112,36,'#BDD5EB')
node('policy','联合配置策略',194,61,126,36,'#BDD5EB')
node('action','区域配水动作',394,61,112,36,'#BDD5EB')
node('transition','环境状态转移',394,147,112,36,'#F1D9B5')
node('end','完成81日',194,138,126,54,'#FFFFFF',True)
node('reward','终点奖励',8,147,112,36,'#E3ECE2')
node('weather','当日天气',394,209,112,32,'#F1D9B5')
def edge(k,source,target,exitxy,entryxy,label='',points=None):
 st='edgeStyle=none;rounded=0;html=0;strokeColor=#527492;strokeWidth=0.8;endArrow=block;endFill=1;endSize=5;fontFamily=Times New Roman,SimSun;fontSize=14;labelBackgroundColor=#ffffff;'+f'exitX={exitxy[0]};exitY={exitxy[1]};entryX={entryxy[0]};entryY={entryxy[1]};'
 c=ET.SubElement(root,'mxCell',{'id':k,'value':label,'style':st,'edge':'1','parent':'1','source':source,'target':target})
 geo=ET.SubElement(c,'mxGeometry',{'as':'geometry','relative':'1'})
 if points:
  arr=ET.SubElement(geo,'Array',{'as':'points'})
  for x,y in points:ET.SubElement(arr,'mxPoint',{'x':str(x),'y':str(y)})
edge('initialize-observe','initialize','observe',(0.5,1),(0.5,0))
edge('observe-policy','observe','policy',(1,0.5),(0,0.5))
edge('policy-action','policy','action',(1,0.5),(0,0.5))
edge('action-transition','action','transition',(0.5,1),(0.5,0))
edge('weather-transition','weather','transition',(0.5,0),(0.5,1))
edge('transition-end','transition','end',(0,0.5),(1,0.5))
edge('end-observe','end','observe',(0.5,0),(0.5,1),'否',[(257,117),(64,117)])
edge('end-reward','end','reward',(0,0.5),(1,0.5),'是')
ET.indent(mx,space='  ')
ET.ElementTree(mx).write(O/'source.drawio',encoding='utf-8',xml_declaration=True)
print('Generated editable source.drawio; export it with the thesis scripts/export_method_figures.py tool.')
