#!/usr/bin/env python3
"""Render the current saved learning curves as one row of two native panels.

The existing local CSV renderer supplies every point and standard-deviation
band. Its original 59.2x51mm axes match Chapter5. Only the measured common
legend gap changes from8bp to15bp; no model or environment is imported.
"""
from pathlib import Path
import argparse, hashlib, json, shutil, sys, tempfile, types
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



# RL terminology revision 2026-11-14: final text layer after original guards.
RL_WORDING_MAP = [('训练步数', '训练环境交互步数')]
RL_WORDING_AXIS = None

def _rl_plain(v):
    import numpy as np
    if isinstance(v,np.ndarray):return _rl_plain(v.tolist())
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,(tuple,list)):return [_rl_plain(x) for x in v]
    if isinstance(v,dict):return {str(k):_rl_plain(x) for k,x in v.items()}
    return v

def _rl_scientific(fig):
    return _rl_plain(dict(canvas=fig.get_size_inches(),axes=[dict(
        position=ax.get_position().bounds,xlim=ax.get_xlim(),ylim=ax.get_ylim(),
        xscale=ax.get_xscale(),yscale=ax.get_yscale(),xticks=ax.get_xticks(),yticks=ax.get_yticks(),
        lines=[dict(x=x.get_xdata(),y=x.get_ydata(),color=x.get_color(),width=x.get_linewidth(),
                    style=x.get_linestyle(),marker=x.get_marker(),ms=x.get_markersize(),
                    alpha=x.get_alpha(),zorder=x.get_zorder())for x in ax.lines],
        collections=[dict(paths=[dict(vertices=p.vertices,codes=p.codes)for p in x.get_paths()],
                          offsets=x.get_offsets(),array=x.get_array(),clim=x.get_clim(),
                          fc=x.get_facecolors(),ec=x.get_edgecolors(),lw=x.get_linewidths(),
                          alpha=x.get_alpha(),zorder=x.get_zorder())for x in ax.collections],
        patches=[dict(vertices=x.get_path().vertices,codes=x.get_path().codes,
                      transform=x.get_patch_transform().get_matrix(),fc=x.get_facecolor(),
                      ec=x.get_edgecolor(),lw=x.get_linewidth(),alpha=x.get_alpha(),zorder=x.get_zorder())for x in ax.patches],
        images=[dict(array=x.get_array(),extent=x.get_extent(),clim=x.get_clim())for x in ax.images]
        )for ax in fig.axes]))

def _rl_export_wording(base,fig,target,*args,**kwargs):
    from matplotlib.text import Text
    from matplotlib.ticker import FuncFormatter
    import json,hashlib
    from pathlib import Path
    try:path=Path(target)
    except TypeError:return base(fig,target,*args,**kwargs)
    if path.name not in ('figure.pdf','figure.svg','figure.png'):
        return base(fig,target,*args,**kwargs)
    fig.canvas.draw();before=_rl_scientific(fig)
    artists=[t for t in fig.findobj(Text)if t.get_visible()and t.get_text()]
    fonts=lambda:[(id(t),t.get_fontsize(),t.get_fontfamily(),t.get_fontweight(),
                   t.get_fontstyle(),t.get_position(),t.get_rotation(),t.get_ha(),t.get_va())for t in artists]
    font_before=fonts();changed=[];formatters=[];axis_records=[]
    try:
        if RL_WORDING_AXIS:
            ax=fig.axes[RL_WORDING_AXIS['axis_index']]
            formatters.append((ax.xaxis,ax.xaxis.get_major_formatter()))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:format(v/RL_WORDING_AXIS['divisor'],'g')))
            old=ax.get_xlabel();new=RL_WORDING_AXIS['xlabel']
            changed.append((ax.xaxis.label,old,new));ax.xaxis.label.set_text(new)
            axis_records.append(dict(axis_index=RL_WORDING_AXIS['axis_index'],
                                     numeric_ticks_before=list(ax.get_xticks()),divisor=RL_WORDING_AXIS['divisor']))
        for t in artists:
            old=t.get_text();new=old
            for a,b in RL_WORDING_MAP:new=new.replace(a,b)
            if new!=old:changed.append((t,old,new));t.set_text(new)
        assert changed,'Expected final wording labels were not found'
        fig.canvas.draw()
        assert _rl_scientific(fig)==before,'Non-text artists, axes, ticks or canvas changed'
        assert fonts()==font_before,'Font, rotation or text anchor changed'
        ren=fig.canvas.get_renderer();crop=kwargs.get('bbox_inches')
        bounds=fig.bbox
        if crop is not None and hasattr(crop,'bounds'):
            from matplotlib.transforms import Bbox
            bounds=Bbox.from_bounds(*(x*fig.dpi for x in crop.bounds))
        for t,old,new in changed:
            bb=t.get_window_extent(ren)
            assert bb.x0>=bounds.x0-1 and bb.x1<=bounds.x1+1 and bb.y0>=bounds.y0-1 and bb.y1<=bounds.y1+1,('Reflow wording; do not shrink fonts',new,bb.bounds,bounds.bounds)
        result=base(fig,target,*args,**kwargs)
        assert _rl_scientific(fig)==before,'Save changed numerical or geometric artists'
        proof=dict(schema='rl-wording-final-native-label-layer-v1',status='PASS',
                   original_render_guards_retained=True,only_final_labels_changed=True,
                   numeric_artists_axes_canvas_fonts_exact=True,
                   scientific_before_sha256=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),
                   scientific_after_sha256=hashlib.sha256(json.dumps(_rl_scientific(fig),sort_keys=True).encode()).hexdigest(),
                   labels=[dict(before=old,after=new,font_bp=t.get_fontsize(),bbox_pixels=list(t.get_window_extent(ren).bounds))for t,old,new in changed],
                   axis_formatter=axis_records,environment_steps=0,model_loads=0,policy_inferences=0,training_runs=0)
        (path.parent/'rl_wording_label_proof.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
        return result
    finally:
        for axis,formatter in formatters:axis.set_major_formatter(formatter)
        for t,old,new in reversed(changed):t.set_text(old)

def _rl_capture_final(directory):
    from matplotlib.figure import Figure
    from pathlib import Path
    base=Figure.savefig;captured={}
    def capture(fig,target,*args,**kwargs):
        try:path=Path(target)
        except TypeError:return base(fig,target,*args,**kwargs)
        if path.name in ('figure.pdf','figure.svg','figure.png')and path.parent.resolve()==Path(directory).resolve():
            captured[path.name]=(fig,args,dict(kwargs))
        return base(fig,target,*args,**kwargs)
    Figure.savefig=capture
    return base,captured

def _rl_restore_capture(base):
    from matplotlib.figure import Figure
    Figure.savefig=base

def _rl_emit_captured(base,captured,directory):
    from pathlib import Path
    assert 'figure.pdf'in captured and 'figure.png'in captured,'No current final native figure captured'
    for name,(fig,args,kwargs)in captured.items():
        _rl_export_wording(base,fig,Path(directory)/name,*args,**kwargs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=HERE)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = HERE / 'source/spacing.py'
    substitutions = [
        ("move_legend(len(fig.legends)-1,8-f['xlabel_to_legend_bp'])",
         "move_legend(len(fig.legends)-1,15-f['xlabel_to_legend_bp'])"),
        ("abs(final['xlabel_to_legend_bp']-8)",
         "abs(final['xlabel_to_legend_bp']-15)"),
    ]
    with tempfile.TemporaryDirectory(prefix='.horizontal15-', dir=out) as tmp:
        work = Path(tmp)
        code = source.read_text()
        for old, new in substitutions:
            assert code.count(old) == 1, old
            code = code.replace(old, new)
        module = types.ModuleType('c6_horizontal_spacing15')
        module.__file__ = str(source)
        exec(compile(code, str(source) + '[horizontal15-memory]', 'exec'), module.__dict__)
        module.ROOT = work
        rendered = work / 'rendered'
        _rl_base_save,_rl_captured=_rl_capture_final(rendered/'accepted_stage'/'figure')
        old_argv = sys.argv[:]
        try:
            sys.argv = [__file__, '--number', '6.5', '--output-dir', str(rendered)]
            module.main()
        finally:
            sys.argv = old_argv
            _rl_restore_capture(_rl_base_save)
        values = json.loads((rendered / 'numeric_artist_values_before_after.json').read_text())
        parent = json.loads((HERE / 'narrow15_numeric_artists.json').read_text())
        assert values['before'] == values['after'] == parent['after']
        geometry = json.loads((rendered / 'render_geometry.json').read_text())
        assert abs(geometry['native_mm'][0] - 156) < 1e-8
        assert all(abs(w - 59.2) < 1e-8 and abs(h - 51) < 1e-8
                   for w, h in geometry['axes_sizes_mm'])
        assert abs(geometry['xlabel_to_legend_bp'] - 15) < .002
        assert geometry['all_text_inside_canvas']
        _rl_emit_captured(_rl_base_save,_rl_captured,rendered)
        shutil.copyfile(rendered/'rl_wording_label_proof.json',out/'rl_wording_label_proof.json')
        for ext in ('pdf', 'svg', 'png'):
            shutil.copyfile(rendered / f'figure.{ext}', out / f'horizontal15.{ext}')
        record = {
            'schema': 'chapter6-horizontal-learning15-v1',
            'status': 'PASS_REQUIRES_VISUAL_REVIEW',
            'native_mm': geometry['native_mm'],
            'axes_mm': geometry['axes_sizes_mm'],
            'font_bp': 10.5,
            'shared_legend_gap_bp': geometry['xlabel_to_legend_bp'],
            'geometry': geometry,
            'numeric_artists_equal_current_parent': True,
            'parent_numeric_record_sha256': sha(HERE / 'narrow15_numeric_artists.json'),
            'CSV_sha256': {p.name: sha(p) for p in HERE.glob('*.csv')},
            'renderer': Path(__file__).name,
            'renderer_sha256': sha(__file__),
            'unchanged_local_spacing_source_sha256': sha(source),
            'output_sha256': {f'horizontal15.{e}': sha(out / f'horizontal15.{e}')
                              for e in ('pdf', 'svg', 'png')},
            'scientific_execution': 0,
        }
        (out / 'horizontal15_geometry.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        (out / 'horizontal15_numeric_artists.json').write_text(json.dumps(values, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'status': record['status'], 'native_mm': record['native_mm'],
                          'axes_mm': record['axes_mm']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
