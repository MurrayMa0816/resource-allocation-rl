#!/usr/bin/env python3
"""Search the saved paper-to-experiment catalogue without loading scientific code."""
from pathlib import Path
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chapter', type=int, choices=range(2, 7))
    parser.add_argument('--section', help='例如 4.3.3.2')
    parser.add_argument('--figure', help='例如 4.8')
    parser.add_argument('--table', help='例如 3.3')
    parser.add_argument('--model', help='模型ID或文件名中的文字')
    parser.add_argument('--method', help='论文方法名称或实现家族')
    parser.add_argument('--seed', help='随机种子')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    catalogue = json.loads((root / 'registry/paper_navigation.json').read_text())
    chapters = catalogue['chapters']
    if args.chapter:
        chapters = [c for c in chapters if c['chapter'] == f'chapter{args.chapter:02d}']
    sections = [dict(s, chapter=c['chapter']) for c in chapters for s in c['sections']]
    assets = [dict(a, chapter=c['chapter']) for c in chapters for a in c['assets']]
    models = [dict(m, chapter=c['chapter']) for c in chapters for m in c['models']]
    if args.section:
        sections = [s for s in sections if s['number'] == args.section or s['number'].startswith(args.section + '.')]
        models = [m for m in models if any(s == args.section or s.startswith(args.section + '.') for s in m['paper_sections'])]
        assets = [a for a in assets if any(s == args.section or s.startswith(args.section + '.') for s in a['paper_sections'])]
    for kind, number in [('figure', args.figure), ('table', args.table)]:
        if number:
            number = number.removeprefix('图').removeprefix('表')
            assets = [a for a in assets if a['kind'] == kind and a['number'] == number]
            ids = {model_id for a in assets for model_id in a['model_ids']}
            models = [m for m in models if m['id'] in ids]
            numbers = {s for a in assets for s in a['paper_sections']}
            sections = [s for s in sections if s['number'] in numbers]
    if args.model:
        query = args.model.casefold()
        models = [m for m in models if query in (m['id'] + ' ' + m['model']).casefold()]
    if args.method:
        query = args.method.casefold()
        exact = [m for m in models if query == m['paper_method'].casefold()]
        human = [m for m in models if query in m['paper_method'].casefold()]
        models = exact or human or [m for m in models if query in str(m.get('method') or m.get('variant') or '').casefold()]
    if args.seed is not None:
        models = [m for m in models if str(m.get('seed', '')) == args.seed]
    if args.model or args.method or args.seed is not None:
        ids = {m['id'] for m in models}
        assets = [a for a in assets if ids.intersection(a['model_ids'])]
        uses = {s for m in models for s in m['paper_sections']}
        sections = [s for s in sections if any(u == s['number'] or u.startswith(s['number'] + '.') for u in uses)]
    querying = any([args.section, args.figure, args.table, args.model, args.method, args.seed is not None])
    result = {'data_root': str(root), 'sections': sections,
              'models_found': len(models), 'model_loading': 0, 'scientific_execution': 0}
    if querying:
        result['assets'] = assets
        result['selected_actual_inputs'] = [
            dict(kind=a['kind'], number=a['number'], chapter=a['chapter'],
                 evaluation_identity=a['evaluation_identity'],
                 actual_inputs=a['actual_inputs'],
                 actual_evaluation_ids=a.get('actual_evaluation_ids', []),
                 context_evaluation_ids=a.get('context_evaluation_ids', []),
                 actual_DEV_model_ids=a.get('actual_DEV_model_ids', []),
                 context_evaluation_identity=a.get('context_evaluation_identity', ''))
            for a in assets]
        result['record_semantics'] = {
            'selected_actual_inputs': '本次所选图表实际读取的本地图包数据、配置与源码；评价身份单独注明。',
            'models': '相关保留模型的身份摘要，不把其全部DEV/TEST列为图表实际输入。',
            'models_full_record_catalogue': '模型附带的完整训练/DEV/TEST与源码记录；保留全部原记录，可能涉及其他小节或补充评价。'}
        result['models_full_record_catalogue'] = [
            dict(m, absolute_model=str(root / m['model']),
                 record_role='complete_attached_records_not_asset_actual_inputs')
            for m in models]
        excluded = {'training_paths', 'dev_paths', 'test_paths', 'config_paths',
                    'code_paths', 'diagnostic_paths', 'section_uses'}
        result['models'] = [
            dict({key: value for key, value in m.items() if key not in excluded},
                 absolute_model=str(root / m['model']),
                 complete_record_catalogue_id=m['id'])
            for m in models]
    else:
        result['usage'] = ['python3 lookup.py --figure 4.8',
                           'python3 lookup.py --section 5.3.2.2',
                           'python3 lookup.py --chapter 6 --method SeDVal --seed 6400']
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
