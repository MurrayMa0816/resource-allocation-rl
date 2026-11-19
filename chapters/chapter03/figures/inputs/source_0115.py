"""Pure-array and PDF audit; does not import a renderer, model or environment."""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import numpy as np
import fitz

asset = Path(__file__).resolve().parent.parent
parent = Path('/hy-tmp/resource_allocation_results/thesis_print_format_implementation_20261109_160917/figures/assets/figure3-6')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(path):
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt', newline='') as stream:
        return list(csv.DictReader(stream))


raw = read(asset/'inputs/data.csv.gz')
selected = read(asset/'display_seed_data.csv')
aggregates = read(asset/'display_aggregates.csv')
parent_aggregates = read(parent/'display_aggregates.csv.gz')
assert sha(asset/'inputs/data.csv.gz') == sha(parent/'inputs/data.csv.gz') == '15539369343c4ba35f5db94401304cb4323cd6e05ddbdd11ffcbcce5ffc6a88b'
assert len(raw) == 250 and len(selected) == 150 and len(aggregates) == 36
for row in aggregates:
    assert row in parent_aggregates
external = {'radio': 'c_ppo', 'v5': 'gradient_ppo'}
assert selected == [r for r in raw if r['method'] in ['A_FULL', 'A_PACE', external[r['domain']]]]
artists = json.loads((asset/'audit/retained_artists.json').read_text())
panels = [('radio', ['budget08', 'default', 'budget12'], [384, 480, 576]),
          ('radio', ['persistence02', 'default', 'persistence095'], [20, 80, 95]),
          ('v5', ['budget08', 'default', 'budget12'], [16128, 20160, 24192]),
          ('v5', ['default', 'arrivals_front', 'arrivals_middle'], [0, 1, 2])]
labels = {'A_FULL': 'ReJoP', 'A_PACE': '去总量学习',
          'c_ppo': 'C-PPO（通信）/Gradient PPO（柴油）',
          'gradient_ppo': 'C-PPO（通信）/Gradient PPO（柴油）'}
for index, (domain, conditions, xx) in enumerate(panels):
    for method in ['A_FULL', 'A_PACE', external[domain]]:
        means, sds = [], []
        for condition in conditions:
            records = sorted([r for r in raw if r['domain'] == domain and r['condition'] == condition
                              and r['method'] == method], key=lambda r: int(r['seed']))
            assert [int(r['seed']) for r in records] == list(range(5))
            assert {int(r['paths']) for r in records} == {128}
            values = np.array([float(r['score_mean']) for r in records])
            means.append(values.mean())
            sds.append(values.std(ddof=1))
        item = artists[index][labels[method]]
        assert np.array_equal(item['x'], xx) and np.array_equal(item['y'], means)
        expected = np.stack([[[x, mean-sd], [x, mean+sd]] for x, mean, sd in zip(xx, means, sds)])
        assert np.array_equal(item['errorbar_segments'][0], expected)
        assert np.array_equal(np.array(item['caps'])[0], [[x, mean-sd] for x, mean, sd in zip(xx, means, sds)])
        assert np.array_equal(np.array(item['caps'])[1], [[x, mean+sd] for x, mean, sd in zip(xx, means, sds)])
proof = json.loads((asset/'audit/presentation_verification.json').read_text())
original = json.loads((asset/'inputs/artist_values.json').read_text())
for actual, inherited in zip(proof['geometry_after'], original['science']):
    for key in ['position', 'xlim', 'ylim']:
        assert actual[key] == inherited[key]
assert proof['geometry_before'] == proof['geometry_after']
document = fitz.open(asset/'figure.pdf')
assert len(document) == 1
page = document[0]
spans = [span for block in page.get_text('dict')['blocks'] if 'lines' in block
         for line in block['lines'] for span in line['spans'] if span['text'].strip()]
assert all(abs(span['size']-10.5) < 1e-5 for span in spans)
assert abs(page.rect.width/72*25.4-156) < 1e-4 and abs(page.rect.height/72*25.4-157) < 1e-4
assert '直接预测' not in page.get_text() and '仅看资源' not in page.get_text() and '去总量学习' in page.get_text()
assert sorted({font[3].split('+')[-1] for font in page.get_fonts()}) == ['SimSun', 'TimesNewRomanPSMT']
report = dict(status='PASS',
              review_scope='Separate pure-array and PDF checks, performed by figure author; not a second agent review',
              parent_250_seed_rows_sha256=sha(asset/'inputs/data.csv.gz'),
              retained_150_seed_rows_exact=True,
              displayed_36_aggregate_rows_exact_parent_members=True,
              retained_12_series_and_all_36_mean_sd_errorbar_nodes_recomputed_exact=True,
              axes_position_xlim_ylim_exact_parent=True, axes_ticks_exact_before_after=True,
              pdf_pages=1, pdf_dimensions_mm=[page.rect.width/72*25.4, page.rect.height/72*25.4],
              all_nonempty_pdf_text_font_bp=10.5, fonts=['SimSun', 'Times New Roman'],
              new_environment_steps=0, model_loads=0, policy_inferences=0, training_runs=0, thesis_builds=0)
(asset/'audit/numeric_and_pdf_author_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(report, ensure_ascii=False))
