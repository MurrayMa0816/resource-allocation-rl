from pathlib import Path
import importlib.util,math,json,hashlib
import fitz

def compose(output,maps,package):
    R=Path(output);A=Path(package)
    spec=importlib.util.spec_from_file_location('local_display',A/'source/display_helpers.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);plt=m.plotting(R)
    from matplotlib.patches import Circle,Rectangle
    import matplotlib.patheffects as pe
    # The native legend keeps actual10.5bp letters, and scales physical circle
    # radii by the same110/156 geometry factor as the two native maps.
    LW,LH=110.,21.5;fig=plt.figure(figsize=(LW/25.4,LH/25.4));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,LW),ylim=(0,LH));ax.axis('off')
    ax.text(55,19.1,'日初储水深度（mm）',ha='center',va='center',fontsize=10.5)
    left,right=5*110/156,151*110/156;step=(right-left)/5
    for i,c in enumerate(['#fff9df','#ffedc5','#ffdc9c','#ffc775','#f6ac52']):ax.add_patch(Rectangle((left+i*step,14.8),step,1.6,fc=c,ec='none'))
    ax.add_patch(Rectangle((left,14.8),right-left,1.6,fc='none',ec='#65725e',lw=.8))
    for i,x in enumerate([80,100,120,140,160,180]):ax.text(left+i*step,12.0,str(x),ha='center',va='center',fontsize=10.5)
    ax.text(left,4.8,'当日配水量\n（m³）',ha='left',va='center',fontsize=10.5,linespacing=1.05)
    for x,v in [(31.,125),(61.,1000),(94.5,4000)]:
     radius=1.6*110/156*math.sqrt(v/1000);c=Circle((x,4.8),radius,fc='none',ec='#A80058',lw=1.5);c.set_path_effects([pe.Stroke(linewidth=3.,foreground='white'),pe.Normal()]);ax.add_patch(c)
     ax.text(right if v==4000 else x+radius+1.3,4.8,str(v),ha='right'if v==4000 else'left',va='center',fontsize=10.5)
    fig.savefig(R/'legend.pdf',metadata={'CreationDate':None,'ModDate':None});fig.savefig(R/'legend.png',dpi=240);plt.close(fig)
    for label,days,titles in [('ab',[0,46],['(a) 5月30日（返青首日）','(b) 7月15日（孕穗首日）']),('cd',[62,72],['(c) 7月31日（抽穗首日）','(d) 8月10日（乳熟首日）'])]:
     W,H=156.,225.;fig=plt.figure(figsize=(W/25.4,H/25.4));fig.text(.5,1-2.8/H,titles[0],ha='center',va='center',fontsize=10.5);fig.text(.5,1-102.8/H,titles[1],ha='center',va='center',fontsize=10.5);fig.savefig(R/f'titles_{label}.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
     doc=fitz.open(R/f'titles_{label}.pdf');page=doc[0]
     for day,y in zip(days,[6.8,106.8]):
      f=fitz.open(maps/f'day{day:02}.pdf');p=f[0];assert abs(p.rect.width*25.4/72-110)<.001
      height=p.rect.height*25.4/72
      page.show_pdf_page(fitz.Rect(23/25.4*72,y/25.4*72,133/25.4*72,(y+height)/25.4*72),f,0)
     f=fitz.open(R/'legend.pdf');page.show_pdf_page(fitz.Rect(23/25.4*72,201.5/25.4*72,133/25.4*72,223/25.4*72),f,0)
     doc.save(R/f'pair_{label}.pdf',garbage=4,deflate=True,no_new_id=True)
     doc.close()
     with fitz.open(R/f'pair_{label}.pdf')as final:
      final[0].get_pixmap(matrix=fitz.Matrix(2.5,2.5),alpha=False).save(R/f'pair_{label}.png')
      (R/f'pair_{label}.svg').write_text(final[0].get_svg_image(text_as_path=False))
     print(label,W,H)
