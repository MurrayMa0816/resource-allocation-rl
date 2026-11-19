"""Offline rendering of Chapter 5's accepted supervision boxplot.

Only the outer canvas and shared legend are reflowed to the thesis PDF-text
spacing standard.  The accepted box statistics, raw observations, axes and
artist styles remain unchanged.  No learner, environment or checkpoint is used.
"""
from pathlib import Path
import argparse, csv, hashlib, io, json, os, re, shutil, sys

sys.dont_write_bytecode = True
ASSET = Path('/hy-tmp/resource_allocation_results/chapter05/paper_revisions/integrate_supervision_boxplot_20261110_171410/assets/figure56_supervision')
PARENT = Path('/hy-tmp/resource_allocation_results/chapter05/figures/value_supervision_boxplot_20261110_161215')
ORIGINAL = Path('/hy-tmp/resource_allocation_results/chapter05/experiments/value_supervision_diagnosis/diagnostic')
THESIS = Path('/root/gamewar/latex_thesis_resource_allocation')
MAINTAINED = Path(__file__).resolve()
WRAPPER = MAINTAINED.parent/'figure_5_6_supervision_boxplot__plot.py'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(p, value):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    out = (args.output_dir or ASSET).resolve()
    assert out.is_relative_to('/hy-tmp/resource_allocation_results')
    out.mkdir(parents=True, exist_ok=True)
    for name, leaf in [('MPLCONFIGDIR', 'mpl'), ('XDG_CACHE_HOME', 'xdg'), ('TMPDIR', 'tmp')]:
        p = out/'runtime'/leaf
        p.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(p)
    inputs = out/'inputs'
    inputs.mkdir(exist_ok=True)
    input_manifest = {}
    for name in ['per_seed_metrics.csv', 'summary.json', 'protocol.json']:
        src = ORIGINAL/name
        dst = inputs/name
        if dst.exists():
            assert sha(src) == sha(dst), name
        else:
            shutil.copyfile(src, dst)
        input_manifest[name] = {'parent': str(src), 'sha256': sha(src)}
    parent_hashes = {name: sha(PARENT/name) for name in ['plot.py', 'source_binding.json', 'plotted_values.json', 'value_supervision_boxplot.pdf', 'value_supervision_boxplot.png', 'value_supervision_boxplot.svg']}
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager, cbook
    from matplotlib.patches import Patch
    from matplotlib.text import Text
    from PIL import Image
    import fitz
    fonts = {p.name.lower():p for p in (THESIS/'assets/fonts/windows').iterdir() if p.is_file()}
    for name in ['times.ttf', 'timesi.ttf', 'simsun.ttc']:
        font_manager.fontManager.addfont(str(fonts[name]))
    plt.rcParams.update({
        'font.family':['Times New Roman', 'SimSun'], 'font.size':10.5,
        'mathtext.fontset':'stix', 'axes.unicode_minus':False,
        'pdf.fonttype':42, 'ps.fonttype':42, 'axes.linewidth':.8,
        'xtick.direction':'out', 'ytick.direction':'out',
    })
    # CM mathtext's period glyph has a legacy colon character code. Preserve
    # the glyph and every coordinate, but give that exact glyph its Unicode dot.
    cm_math_font = Path(matplotlib.get_data_path())/'fonts/ttf/cmmi10.ttf'
    from fontTools.ttLib import TTFont
    with TTFont(cm_math_font) as font:
        assert font.getBestCmap()[0x3a] == 'period'
    def correct_cm_period_unicode(document):
        changed = []
        fonts_seen = {f[0]:f for page in document for f in page.get_fonts(full=True)}
        for xref, info in fonts_seen.items():
            if info[3].split('+')[-1] != 'Cmmi10':
                continue
            kind, value = document.xref_get_key(xref, 'ToUnicode')
            assert kind == 'xref', info
            mapping_xref = int(value.split()[0])
            original = document.xref_stream(mapping_xref)
            embedded_font = document.extract_font(xref)[3]
            with TTFont(io.BytesIO(embedded_font)) as embedded:
                assert embedded.getBestCmap()[0x3a] == 'period'
            legacy = b'<003a> <003a> [<003a>]'
            correct = b'<003a> <003a> [<002e>]'
            if legacy not in original:
                assert correct in original or b'<003a>' not in original, original
                continue
            assert original.count(legacy) == 1
            document.update_stream(mapping_xref, original.replace(legacy,correct))
            changed.append(dict(font=info[3],glyph='period',character_code_hex='003a',
                                old_Unicode='003a',new_Unicode='002e',
                                embedded_font_cmap_glyph='period',
                                exact_mapping_replacement_count=1))
        return changed

    rows = list(csv.DictReader((inputs/'per_seed_metrics.csv').open()))
    summary = json.loads((inputs/'summary.json').read_text())['means']
    width, height = 156., 85.
    fig = plt.figure(figsize=(width/25.4, height/25.4))
    styles = [('full', '#1f5a8d', -.17), ('scalar_value', '#df8539', .17)]
    data = []
    for panel,(domain,title) in enumerate([('radio','(a) 通信功率分配'), ('v5','(b) 应急柴油分配')]):
        ax = fig.add_axes([(18+75*panel)/width, 23/height, 59.2/width, 51/height])
        ax.set_yscale('log')
        ax.set_ylim(.001, 2)
        ax.set_xlim(-.55, 2.55)
        ax.set_yticks([.001, .01, .1, 1])
        ax.minorticks_off()
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', color='#dddddd', linewidth=.5)
        ax.set_axisbelow(True)
        ax.set_title(title, pad=5)
        for arm,color,offset in styles:
            selected = sorted([r for r in rows if r['domain']==domain and r['arm']==arm], key=lambda r:int(r['seed']))
            assert [int(r['seed']) for r in selected] == list(range(5))
            for i,metric in enumerate(['C','J','F']):
                field = 'all_active_steps__mse_'+metric
                values = np.array([float(r[field]) for r in selected])
                mean, sd = float(values.mean()), float(values.std(ddof=1))
                assert (values > .001).all() and (values < 2).all()
                assert abs(mean-summary[domain][arm][field]['mean']) < 1e-14
                assert abs(sd-summary[domain][arm][field]['sd']) < 1e-14
                stats = cbook.boxplot_stats(values, whis=1.5)[0]
                pos = i+offset
                ax.bxp([stats], positions=[pos], widths=.26, showfliers=False, patch_artist=True, manage_ticks=False,
                       boxprops={'facecolor':color+'55','edgecolor':color,'linewidth':1.1},
                       medianprops={'color':color,'linewidth':1.8},
                       whiskerprops={'color':color,'linewidth':.9},
                       capprops={'color':color,'linewidth':.9})
                x = pos+np.linspace(-.045,.045,5)
                ax.plot(x, values, linestyle='none', marker='_', markersize=3.2, markeredgewidth=.85, color='#222222', alpha=.85, zorder=5)
                data.append({'domain':domain,'arm':arm,'metric':metric,'seeds':list(range(5)),'values':values.tolist(),
                             'arithmetic_mean_not_box_center':mean,'sample_SD_not_plotted':sd,
                             'q1':float(stats['q1']),'median':float(stats['med']),'q3':float(stats['q3']),
                             'IQR':float(stats['iqr']),'whisker_low':float(stats['whislo']),'whisker_high':float(stats['whishi']),
                             'fliers_all_shown_as_raw_ticks':stats['fliers'].tolist(),
                             'raw_tick_x':x.tolist(),'quartiles_calculated_before_axis_log_transform':True})
        ax.set_xticks([0,1,2], ['覆盖预测\n$C$','成效预测\n$J$','加权预测\n$F_{0.8}$'], fontfamily='SimSun')
    fig.text(1.5/width,48.5/height,'预测均方误差（越低越好）',rotation=90,va='center',ha='left')
    legend = fig.legend(handles=[Patch(facecolor=c+'55',edgecolor=c,label=l) for (_,c,_),l in zip(styles,['SeDVal（分项监督）','标量监督'])], loc='lower center',bbox_to_anchor=(.5,1.8/height),ncol=2,frameon=False,handlelength=1.25,columnspacing=1.8,borderaxespad=0)
    accepted = json.loads((PARENT/'plotted_values.json').read_text())
    assert data == accepted['data'], 'Accepted box statistics or raw observations changed'

    def plain(value):
        if isinstance(value,np.ndarray): return plain(value.tolist())
        if isinstance(value,np.generic): return value.item()
        if isinstance(value,(tuple,list)): return [plain(v) for v in value]
        if isinstance(value,dict): return {str(k):plain(v) for k,v in value.items()}
        return value

    def scientific():
        return [dict(xlim=plain(ax.get_xlim()),ylim=plain(ax.get_ylim()),yscale=ax.get_yscale(),
                     xticks=plain(ax.get_xticks()),yticks=plain(ax.get_yticks()),
                     title=ax.get_title(),title_font_bp=ax.title.get_fontsize(),title_pad_bp=5,
                     xticklabels=[(t.get_text(),t.get_fontsize()) for t in ax.get_xticklabels()],
                     lines=[dict(x=plain(l.get_xdata()),y=plain(l.get_ydata()),color=l.get_color(),linewidth=l.get_linewidth(),linestyle=l.get_linestyle(),marker=l.get_marker(),markersize=l.get_markersize(),markeredgewidth=l.get_markeredgewidth(),alpha=l.get_alpha(),zorder=l.get_zorder()) for l in ax.lines],
                     boxes=[dict(vertices=plain(p.get_path().vertices),codes=plain(p.get_path().codes),facecolor=plain(p.get_facecolor()),edgecolor=plain(p.get_edgecolor()),linewidth=p.get_linewidth()) for p in ax.patches]) for ax in fig.axes]

    fig.canvas.draw()
    buf = io.BytesIO()
    fig.savefig(buf,format='png',dpi=400)
    native = Image.open(buf).convert('RGBA')
    parent_png = Image.open(PARENT/'value_supervision_boxplot.png').convert('RGBA')
    assert native.size == parent_png.size and np.array_equal(np.asarray(native),np.asarray(parent_png)), 'Accepted preview not reproduced'
    (out/'native_parent_reproduction.png').write_bytes(buf.getvalue())
    native_before = scientific()
    before_positions = [np.asarray(ax.get_position().bounds)*[width,height,width,height] for ax in fig.axes]

    def norm(text):
        text = re.sub(r'\\(?:mathdefault|mathrm|mathregular)', '', text)
        text = text.replace('$','').replace('{','').replace('}','').replace('_','').replace('^','')
        return ''.join(text.split()).replace('−','-')

    def measure():
        fig.canvas.draw()
        buf=io.BytesIO()
        fig.savefig(buf,format='pdf',metadata={'CreationDate':None,'ModDate':None})
        doc=fitz.open(stream=buf.getvalue(),filetype='pdf')
        correct_cm_period_unicode(doc)
        page=doc[0]; H=page.rect.height
        lines=[]
        for block in page.get_text('rawdict')['blocks']:
            for line in block.get('lines',[]):
                chars=[c for span in line['spans'] for c in span.get('chars',[]) if not c['c'].isspace()]
                lines.append((''.join(c['c'] for c in chars).replace('−','-'),chars))
        def box_for_artist(t):
            b=t.get_window_extent(fig.canvas.get_renderer())
            expected=fitz.Rect(b.x0*72/fig.dpi,H-b.y1*72/fig.dpi,b.x1*72/fig.dpi,H-b.y0*72/fig.dpi)
            found=[]
            for part in [norm(s) for s in t.get_text().split('\n') if norm(s)]:
                candidates=[]
                for text,chars in lines:
                    start=0
                    while part and (start:=text.find(part,start))>=0:
                        bb=fitz.Rect(chars[start]['bbox'])
                        for c in chars[start:start+len(part)]: bb|=fitz.Rect(c['bbox'])
                        dist=(bb.x0+bb.x1-expected.x0-expected.x1)**2+(bb.y0+bb.y1-expected.y0-expected.y1)**2
                        candidates.append((dist,bb));start+=len(part)
                if not candidates:
                    # Math exponents may be separate raw-PDF text lines.
                    fragments = ['10',part[2:]] if re.fullmatch(r'10-?\d+',part) else ['F',part[1:]] if part.startswith('F') and len(part)>1 else []
                    fragment_boxes=[]
                    for fragment in fragments:
                        fragment_candidates=[]
                        for text,chars in lines:
                            start=0
                            while fragment and (start:=text.find(fragment,start))>=0:
                                bb=fitz.Rect(chars[start]['bbox'])
                                for c in chars[start:start+len(fragment)]:bb|=fitz.Rect(c['bbox'])
                                dist=(bb.x0+bb.x1-expected.x0-expected.x1)**2+(bb.y0+bb.y1-expected.y0-expected.y1)**2
                                fragment_candidates.append((dist,bb));start+=len(fragment)
                        assert fragment_candidates, ('Cannot match actual PDF math fragment',t.get_text(),part,fragment)
                        fragment_boxes.append(min(fragment_candidates,key=lambda v:v[0])[1])
                    if fragment_boxes:
                        bb=fitz.Rect(fragment_boxes[0])
                        for b in fragment_boxes[1:]:bb|=b
                        candidates=[(0,bb)]
                assert candidates, ('Cannot match actual PDF character frames',t.get_text(),part)
                found.append(min(candidates,key=lambda v:v[0])[1])
            result=fitz.Rect(found[0])
            for b in found[1:]: result|=b
            return result
        hidden=set()
        for ax in fig.axes:
            for axis,lim in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
                for tick in axis.get_major_ticks():
                    if not min(lim)-1e-9 <= tick.get_loc() <= max(lim)+1e-9:
                        hidden.update([id(tick.label1),id(tick.label2)])
        texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and id(t) not in hidden]
        boxes=[box_for_artist(t) for t in texts]
        content=fitz.Rect(boxes[0])
        for b in boxes[1:]: content|=b
        for ax in fig.axes:
            b=ax.get_window_extent()
            content|=fitz.Rect(b.x0*72/fig.dpi,H-b.y1*72/fig.dpi,b.x1*72/fig.dpi,H-b.y0*72/fig.dpi)
        coord_bottom=max(box_for_artist(t).y1 for ax in fig.axes for t in ax.get_xticklabels() if t.get_visible())
        legend_boxes=[box_for_artist(t) for t in legend.get_texts()]
        legend_box=fitz.Rect(legend_boxes[0])
        for b in legend_boxes[1:]:legend_box|=b
        result={'width_bp':page.rect.width,'height_bp':H,'content_top_bp':content.y0,'content_bottom_bp':content.y1,
                'legend_box_bp':list(legend_box),'xlabel_bottom_bp':coord_bottom,
                'xlabel_to_legend_bp':legend_box.y0-coord_bottom,
                'top_margin_mm':content.y0*25.4/72,'bottom_margin_mm':(H-content.y1)*25.4/72,
                'all_pdf_text_exact':True,'unmatched_texts':[],
                'xtick_label_frames_bp':[dict(text=t.get_text(),bbox_bp=list(box_for_artist(t))) for ax in fig.axes for t in ax.get_xticklabels() if t.get_visible()],
                'PDF_fonts':sorted({span['font'] for block in page.get_text('dict')['blocks'] for line in block.get('lines',[]) for span in line['spans'] if span['text'].strip()}),
                'Chinese_xlabel_frames':[dict(text=span['text'],origin=list(span['origin']),bbox=list(span['bbox'])) for block in page.get_text('dict')['blocks'] for line in block.get('lines',[]) for span in line['spans'] if span['text'] in ['覆盖预测','成效预测','加权预测']]}
        doc.close()
        return result

    native_frames=measure()
    for ax in fig.axes:
        ax.title.set_fontsize(10.5)
    before=scientific()
    for original,normalized in zip(native_before,before):
        permitted=dict(original,title_font_bp=10.5)
        assert permitted==normalized, 'More than the approved title font size changed'
    initial=measure()
    for iteration in range(5):
        f=measure()
        correction_bp=15-f['xlabel_to_legend_bp']
        anchor=legend.get_bbox_to_anchor()
        legend.set_bbox_to_anchor((anchor.x0/fig.bbox.width,(anchor.y0-correction_bp*fig.dpi/72)/fig.bbox.height),transform=fig.transFigure)
        f=measure()
        topcrop=f['content_top_bp']-1.5*72/25.4
        bottomcrop=f['height_bp']-f['content_bottom_bp']-2*72/25.4
        if max(abs(topcrop),abs(bottomcrop),abs(correction_bp))<.0001: break
        positions=[np.asarray(ax.get_position().bounds)*[width,height,width,height] for ax in fig.axes]
        texts=[(t,np.asarray(t.get_position())*[width,height]) for t in fig.texts]
        anchor=legend.get_bbox_to_anchor()
        anchor_bp=(anchor.x0*72/fig.dpi,anchor.y0*72/fig.dpi)
        newheight=height-(topcrop+bottomcrop)*25.4/72
        fig.set_size_inches(width/25.4,newheight/25.4,forward=True)
        for ax,p in zip(fig.axes,positions):
            ax.set_position([p[0]/width,(p[1]-bottomcrop*25.4/72)/newheight,p[2]/width,p[3]/newheight])
        legend.set_bbox_to_anchor((anchor_bp[0]/(width*72/25.4),(anchor_bp[1]-bottomcrop)/(newheight*72/25.4)),transform=fig.transFigure)
        for t,p in texts:
            t.set_position((p[0]/width,(p[1]-bottomcrop*25.4/72)/newheight))
        height=newheight
    final=measure()
    assert abs(final['top_margin_mm']-1.5)<.003, final
    assert abs(final['bottom_margin_mm']-2)<.003, final
    assert abs(final['xlabel_to_legend_bp']-15)<.003, final
    # Preserve the exact historical STIX parent and its completed15bp layout.
    # Change only the math family in the six existing two-line tick labels.
    accepted_stix_frames = dict(final)
    math_labels = [t for ax in fig.axes for t in ax.get_xticklabels() if '$' in t.get_text()]
    assert len(math_labels) == 6
    old_math_labels = [dict(text=t.get_text(),math_fontfamily=t.get_math_fontfamily(),
                           font_bp=t.get_fontsize(),position=list(t.get_position()),
                           linespacing=t._linespacing) for t in math_labels]
    assert all(v['math_fontfamily']=='stix' and v['font_bp']==10.5 for v in old_math_labels)
    for t in math_labels:
        t.set_math_fontfamily('cm')
    initial_cm_frames = measure()
    # Increasing multiline leading moves only the second, mathematical line
    # under top alignment. Calibrate it from actual PDF glyph frames, keeping
    # every Chinese first line, axes and shared legend at their accepted place.
    original_spacing = old_math_labels[0]['linespacing']
    assert all(v['linespacing']==original_spacing for v in old_math_labels)
    required_down_bp = accepted_stix_frames['xlabel_bottom_bp']-initial_cm_frames['xlabel_bottom_bp']
    if abs(required_down_bp) > .0001:
        for t in math_labels:
            t.set_linespacing(original_spacing+.01)
        probe_frames = measure()
        slope = (probe_frames['xlabel_bottom_bp']-initial_cm_frames['xlabel_bottom_bp'])/.01
        assert slope > 0
        adjusted_spacing = original_spacing+required_down_bp/slope
        for t in math_labels:
            t.set_linespacing(adjusted_spacing)
    final = measure()
    assert final['Chinese_xlabel_frames'] == accepted_stix_frames['Chinese_xlabel_frames']
    assert final['legend_box_bp'] == accepted_stix_frames['legend_box_bp']
    assert abs(final['width_bp']-accepted_stix_frames['width_bp'])<1e-7
    assert abs(final['height_bp']-accepted_stix_frames['height_bp'])<1e-7
    assert abs(final['xlabel_to_legend_bp']-15) < .003, final
    assert abs(final['top_margin_mm']-1.5) < .003 and abs(final['bottom_margin_mm']-2) < .003, final
    assert not any('STIX' in name.upper() for name in final['PDF_fonts']), final['PDF_fonts']
    math_proof = dict(status='PASS',change='Six x-axis mathematical labels STIX to Computer Modern',
                     old_labels=old_math_labels,
                     new_labels=[dict(text=t.get_text(),math_fontfamily=t.get_math_fontfamily(),
                                      font_bp=t.get_fontsize(),position=list(t.get_position()),
                                      linespacing=t._linespacing) for t in math_labels],
                     accepted_STIX_parent_PNG_reproduced_exact=True,
                     final_STIX_layout_before_font_change=accepted_stix_frames,
                     initial_CM_layout=initial_cm_frames,final_CM_layout=final,
                     mathematical_line_down_compensation_bp=required_down_bp,
                     Chinese_first_lines_and_legend_coordinates_exact=True,
                     cmmi10_font_sha256=sha(cm_math_font),
                     CM_period_ToUnicode_correction=dict(font='Cmmi10',glyph='period',
                         cmap_glyph_confirmed=True,source_character_code_hex='003a',
                         output_Unicode_hex='002e',glyph_outlines_and_positions_unchanged=True),
                     scientific_execution=0)
    dump(out/'math_font_proof.json',plain(math_proof))
    after=scientific()
    assert before==after, 'Scientific artists or accepted styles changed'
    after_positions=[np.asarray(ax.get_position().bounds)*[width,height,width,height] for ax in fig.axes]
    assert all(np.allclose(a[2:],b[2:],rtol=0,atol=1e-7) for a,b in zip(before_positions,after_positions))
    fig.canvas.draw()
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip():
            b=t.get_window_extent(fig.canvas.get_renderer())
            assert b.x0>=-.8 and b.y0>=-.8 and b.x1<=fig.bbox.x1+.8 and b.y1<=fig.bbox.y1+.8,(t.get_text(),b.bounds)
    matplotlib.rcParams['svg.hashsalt']='chapter05-supervision-boxplot-20261110'
    for ext in ['pdf','png','svg']:
        metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else None
        fig.savefig(out/('figure.'+ext),dpi=400,metadata=metadata)
        if ext == 'pdf':
            pdf_path=out/'figure.pdf'
            before_unicode=out/'figure.before_unicode.pdf'
            shutil.copyfile(pdf_path,before_unicode)
            def pdf_snapshot(document):
                font_data={}
                for page in document:
                    for f in page.get_fonts(full=True):
                        font_data[f[3]]=hashlib.sha256(document.extract_font(f[0])[3]).hexdigest()
                return dict(content=[hashlib.sha256(page.read_contents()).hexdigest() for page in document],
                            drawings=[page.get_drawings() for page in document],
                            rgb_3x=[hashlib.sha256(page.get_pixmap(matrix=fitz.Matrix(3,3),alpha=False).samples).hexdigest() for page in document],
                            embedded_fonts=font_data)
            with fitz.open(pdf_path) as document:
                snapshot_before=pdf_snapshot(document)
                rawtext_before=document[0].get_text()
                assert ''.join(rawtext_before.split()).count('F0:8')==2
                streams_before={i:document.xref_stream(i) for i in range(1,document.xref_length()) if document.xref_is_stream(i)}
                unicode_changes=correct_cm_period_unicode(document)
                assert len(unicode_changes)==1
                changed_streams=[i for i,stream in streams_before.items() if document.xref_stream(i)!=stream]
                assert len(changed_streams)==1
                old_mapping=streams_before[changed_streams[0]]
                new_mapping=document.xref_stream(changed_streams[0])
                assert new_mapping==old_mapping.replace(b'<003a> <003a> [<003a>]',b'<003a> <003a> [<002e>]')
                assert snapshot_before==pdf_snapshot(document)
                temporary_pdf=out/'figure.unicode.pdf'
                document.save(temporary_pdf,garbage=4,deflate=True,no_new_id=True)
            temporary_pdf.replace(pdf_path)
            with fitz.open(pdf_path) as document:
                assert snapshot_before==pdf_snapshot(document)
                rawtext_after=document[0].get_text()
                text=''.join(rawtext_after.split())
                assert text.count('F0.8')==2 and 'F0:8' not in text, text
                assert rawtext_after==rawtext_before.replace('0:8','0.8')
            math_proof['CM_period_ToUnicode_correction'].update(dict(
                exact_changes=unicode_changes,changed_decoded_stream_count=1,
                only_exact_mapping_line_changed=True,
                before_pdf='figure.before_unicode.pdf',before_pdf_sha256=sha(before_unicode),
                final_pdf_sha256=sha(pdf_path),
                page_content_streams_sha256=snapshot_before['content'],
                RGB_3x_sha256=snapshot_before['rgb_3x'],
                embedded_font_sha256=snapshot_before['embedded_fonts'],
                rendered_pixels_3x_exact_before_after=True,
                page_content_streams_and_vectors_exact_before_after=True,
                embedded_fonts_exact_before_after=True,
                text_only_two_period_Unicode_corrections=True,
                rawtext_before=rawtext_before,rawtext_after=rawtext_after))
            dump(out/'math_font_proof.json',plain(math_proof))
    plotted=dict(accepted,native_mm=[width,height])
    dump(out/'plotted_values.json',plotted)
    numeric={'status':'PASS','native_preview_before_title_normalization':native_before,'before':before,'after':after,
             'scientific_artists_after_title_normalization_before_after_exact':True,
             'approved_title_font_change_bp':[native_before[0]['title_font_bp'],10.5]}
    dump(out/'numeric_artist_values_before_after.json',numeric)
    proof={'status':'PASS','number':'5.6','native_mm':[width,height],
           'axes_positions_before_mm':plain(before_positions),'axes_positions_after_mm':plain(after_positions),
           'axes_physical_dimensions_exact':True,'all_non_title_scientific_values_and_styles_preserved':True,
           'accepted_parent_PNG_pixel_exact':True,'pdf_frames_native_preview':native_frames,'pdf_frames_initial':initial,'pdf_frames_final':final,
           'measurement_basis':'native PDF character frames matched to all visible text artists, plus unchanged axes frames',
           'ordinary_font_bp':10.5,'accepted_title_font_before_bp':native_before[0]['title_font_bp'],'title_font_after_bp':10.5,
           'approved_title_font_change_scope':'The two subplot titles only, normalized from12.6bp to10.5bp',
           'accepted_title_pad_bp':5,'xlabel_to_legend_bp':final['xlabel_to_legend_bp'],
           'top_content_to_file_top_mm':final['top_margin_mm'],'content_bottom_to_file_bottom_mm':final['bottom_margin_mm'],
           'new_scientific_execution':0}
    dump(out/'spacing_proof.json',proof)
    (out/'plot.py').write_text('from pathlib import Path\nimport hashlib,runpy\np=Path('+repr(str(MAINTAINED))+')\nassert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(sha(MAINTAINED))+'\nrunpy.run_path(str(p),run_name="__main__")\n')
    source={'schema':'chapter5-supervision-boxplot-spacing15-v1','number':'5.6','status':'PASS',
            'renderer':{'path':str(out/'plot.py'),'sha256':sha(out/'plot.py')},
            'maintenance_renderer':{'path':str(MAINTAINED),'sha256':sha(MAINTAINED)},
            'wrapper':{'path':str(WRAPPER),'sha256':sha(WRAPPER)},
            'parent_directory':str(PARENT),'parent_files_sha256':parent_hashes,
            'inputs':input_manifest,'scientific_data_scope':'Existing DEV/validation, realized terminal-return MSE on each policy own visited states',
            'input_copy_sha256':{name:sha(inputs/name) for name in input_manifest},
            'native_mm':[width,height],'axes_mm':[59.2,51],'boxes':12,'raw_seed_observations':60,
            'no_model_environment_training_inference_or_new_evaluation':True,'scientific_execution':0,
            'approved_title_font_change_bp':[native_before[0]['title_font_bp'],10.5],
            'spacing_proof':'spacing_proof.json','data_proof':'plotted_values.json',
            'math_font_correction':dict(family='Computer Modern',proof='math_font_proof.json',six_existing_labels=True),
            'outputs_sha256':{f'figure.{e}':sha(out/f'figure.{e}') for e in ['pdf','png','svg']}}
    dump(out/'source.json',source)
    dump(out/'render_receipt.json',{'status':'PASS','number':'5.6','native_mm':[width,height],
         'outputs_sha256':{name:sha(out/name) for name in ['figure.pdf','figure.png','figure.svg','plotted_values.json','spacing_proof.json','numeric_artist_values_before_after.json','source.json','plot.py']},
         'new_scientific_execution':0})
    print(json.dumps({'status':'PASS','output':str(out),'native_mm':[width,height],'initial':initial,'final':final},ensure_ascii=False))

if __name__=='__main__':
    main()
