"""Final, reversible text-only export layer after the guarded native renderer.

No saved observations, axes, data paths, styles, or fonts are changed. Native
reproduction must pass first. These labels match the user-approved review PDF.
"""
from pathlib import Path
import hashlib,json,os,runpy
import fitz
from PIL import Image

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pixel_sha(page,scale):return hashlib.sha256(page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False).samples).hexdigest()

def finalize(directory,package,font_dir=None):
    directory,package=Path(directory).resolve(),Path(package).resolve()
    layer=package/'wording_layer_20261119'
    config=json.loads((layer/'label_config.json').read_text())
    if font_dir is None:
        thesis=next((p for p in package.parents if (p/'assets/fonts/windows').is_dir()),None)
        font_dir=Path(os.environ.get('THESIS_FIGURE_FONT_DIR','')) if thesis is None else thesis/'assets/fonts/windows'
    font_dir=Path(font_dir)
    fonts={'song':font_dir/'simsun.ttc','times':font_dir/'TIMES.TTF'}
    assert all(sha(fonts[k])==v for k,v in config['font_sha256'].items()),'Changed font inputs'
    source=directory/'figure.pdf'
    doc=fitz.open(source);page=doc[0]
    assert len(doc)==1 and list(page.rect)==config['rect_pt'],'Native page geometry changed'
    if '训练环境交互步数'in page.get_text():
        if sha(source)!=config['native_pdf_sha256']:
            assert all(pixel_sha(page,int(k))==v for k,v in config['native_rgb_sha256'].items()),'Native reproduction no longer matches accepted data/geometry/text'
        for edit in config['edits']:
            assert edit['old']in page.get_text(),'Expected native wording not present'
            page.add_redact_annot(fitz.Rect(edit['old_bbox']),fill=False,cross_out=False)
        page.apply_redactions(images=0,graphics=0,text=0)
        for edit in config['edits']:
            x=edit['new_left']
            for part in edit['parts']:
                page.insert_text((x,edit['baseline']),part['text'],fontsize=10.5,fontname='ReviewSong' if part['font']=='song' else 'ReviewTimes',fontfile=str(fonts[part['font']]),color=(0,0,0))
                x+=part['width']
        doc.subset_fonts()
        temp=directory/'figure.wording-tmp.pdf';doc.save(temp,garbage=4,deflate=True);doc.close();os.replace(temp,source)
    else:
        assert all(pixel_sha(page,int(k))==v for k,v in config['approved_rgb_sha256'].items()),'Existing final output is not the approved review image'
        doc.close()
    with fitz.open(source)as doc:
        page=doc[0]
        for scale,digest in config['approved_rgb_sha256'].items():
            assert pixel_sha(page,int(scale))==digest,'Final PDF differs visibly from user-approved review'
        assert '训练环境交互步数'not in page.get_text()
        if config['figure']=='2.5':assert '比例映射'not in page.get_text()and'按比例分配'in page.get_text()
        pix=page.get_pixmap(matrix=fitz.Matrix(config['png_dpi']/72,config['png_dpi']/72),alpha=False)
        im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
        width,height=config['png_size']
        assert pix.width>=width and pix.height>=height
        # Matplotlib's accepted PNG rounds canvas dimensions down. Keep its DPI
        # and pixel dimensions by removing only the unpainted border pixels.
        for box in [(width,0,pix.width,pix.height),(0,height,pix.width,pix.height)]:
            if box[2]>box[0]and box[3]>box[1]:assert all(lo==hi==255 for lo,hi in im.crop(box).getextrema()),'Nonwhite pixels would be cropped'
        im=im.crop((0,0,width,height)).convert(config['png_mode'])
        im.save(directory/'figure.png',dpi=(config['png_dpi'],config['png_dpi']))
    svg_proof=runpy.run_path(str(layer/'svg_label_export.py'))['patch_svg'](directory/'figure.svg')
    assert svg_proof.get('changed_axis_labels',svg_proof.get('axis_count',config['axis_count']))==config['axis_count']
    receipt={'schema':'experiment-wording-final-export-20261119','figure':config['figure'],'status':'PASS','native_renderer_guards_preserved':True,'native_output_checked_before_final_text_layer':True,'approved_review_rgb_exact_scales':list(map(int,config['approved_rgb_sha256'])),'canvas_fonts_axes_data_unchanged':True,'axis_count':config['axis_count'],'legend_count':int(config['figure']=='2.5'),'svg_proof':svg_proof,'png_size':[width,height],'png_dpi':config['png_dpi'],'png_rendering':'Approved vector PDF, same DPI and canvas pixel dimensions; only unpainted fractional-edge pixels omitted','outputs_sha256':{f'figure.{ext}':sha(directory/f'figure.{ext}')for ext in ('pdf','svg','png')},'scientific_execution':0}
    (directory/'wording_export_20261119.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'wording_layer_20261119':'PASS','figure':config['figure'],'outputs':receipt['outputs_sha256']},ensure_ascii=False),flush=True)
    return receipt
