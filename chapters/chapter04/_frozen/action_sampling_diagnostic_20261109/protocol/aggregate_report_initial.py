"""Recompute the fixed endpoint sampling diagnostic from all case records."""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    plan_path = ROOT/'protocol/PLAN.json'
    seal = json.loads((ROOT/'protocol/plan_seal.json').read_text())
    assert sha(plan_path) == seal['plan_sha256']
    assert sha(ROOT/'protocol/protected_inputs.json') == seal['protected_inputs_sha256']
    protected = json.loads((ROOT/'protocol/protected_inputs.json').read_text())['files']
    for row in protected:
        assert sha(row['path']) == row['sha256'], row['path']
    jobs = []
    case_count = steps = 0
    all_plans = None
    max_recalc_error = max_resource_error = 0.
    per_seed = []
    for method in ('f1', 'props'):
        for seed in range(5):
            directory = ROOT/'jobs'/f'{method}__v5__seed{seed}'
            completed = json.loads((directory/'completed.json').read_text())
            assert not (directory/'failed.json').exists()
            assert completed['status'] == 'complete'
            assert completed['method'] == method and completed['training_seed'] == seed
            assert completed['training_steps'] == completed['independent_TEST_steps'] == 0
            assert completed['DEV_master'] == 202611043101
            assert completed['trajectories'] == 288 and completed['evaluation_steps'] == 6912
            for key in ('model_SHA_unchanged', 'model_tensors_unchanged', 'sources_unchanged',
                        'global_rng_restored', 'all_full_spend', 'all_terminal_only', 'all_horizons_24'):
                assert completed[key], key
            case_path = directory/'case_results.csv.gz'
            assert sha(case_path) == completed['case_results_sha256']
            with gzip.open(case_path, 'rt', newline='') as stream:
                cases = list(csv.DictReader(stream))
            assert len(cases) == 288
            groups = {}
            for row in cases:
                assert row['method'] == method and int(row['training_seed']) == seed
                mode, repeat, index = row['action_mode'], int(row['replicate']), int(row['case_index'])
                group = groups.setdefault((mode, repeat), {})
                assert index not in group
                group[index] = row
                assert int(row['steps']) == 24
                assert row['nonterminal_rewards_zero'] == row['terminal_full_spend_assertion'] == 'True'
                assert float(row['relative_allocation_error']) <= 1e-10
                assert abs(float(row['score_x100']) - float(row['native_return'])*100) < 1e-12
                for key in ('maximum_negative_allocation_violation', 'maximum_per_object_cap_violation',
                            'maximum_completed_object_allocation'):
                    assert float(row[key]) == 0
                max_resource_error = max(max_resource_error, float(row['relative_allocation_error']))
                case_count += 1
                steps += int(row['steps'])
            expected = {('mean', -1)} | {('stochastic', repeat) for repeat in range(8)}
            assert set(groups) == expected
            means = {}
            for key, rows in groups.items():
                assert sorted(rows) == list(range(32))
                plans = [(rows[i]['path_seed'], rows[i]['plan_sha256']) for i in range(32)]
                if all_plans is None:
                    all_plans = plans
                assert plans == all_plans, (method, seed, key)
                means[key] = statistics.mean(float(row['score_x100']) for row in rows.values())
            rounds = {(row['action_mode'], row['replicate']): row for row in completed['all_rounds']}
            assert set(rounds) == expected
            for key, value in means.items():
                error = abs(value - rounds[key]['mean_score_x100'])
                max_recalc_error = max(max_recalc_error, error)
                assert error < 1e-10
                assert rounds[key]['global_rng_restored'] and rounds[key]['model_tensors_unchanged']
                assert rounds[key]['action_seed'] == 91090000 + 1000*seed + max(0, key[1])
            repeats = [means['stochastic', repeat] for repeat in range(8)]
            row = dict(method=method, training_seed=seed, mean_score_x100=means['mean', -1],
                       stochastic_score_x100=statistics.mean(repeats),
                       stochastic_repeat_sd_x100=statistics.stdev(repeats),
                       stochastic_minus_mean_score_points=statistics.mean(repeats) - means['mean', -1],
                       repeat_means_x100=repeats)
            assert abs(row['stochastic_score_x100'] - completed['stochastic_score_x100']) < 1e-10
            assert abs(row['mean_score_x100'] - completed['deterministic_score_x100']) < 1e-10
            per_seed.append(row)
            jobs.append(dict(method=method, training_seed=seed, completion_sha256=sha(directory/'completed.json'),
                             case_results_sha256=sha(case_path)))
    assert case_count == 2880 and steps == 69120
    summaries = {}
    for method in ('f1', 'props'):
        rows = [row for row in per_seed if row['method'] == method]
        summaries[method] = {}
        for mode, field in [('mean', 'mean_score_x100'), ('stochastic', 'stochastic_score_x100')]:
            values = [row[field] for row in rows]
            summaries[method][mode] = dict(mean_score_x100=statistics.mean(values),
                                            training_seed_sample_sd_x100=statistics.stdev(values),
                                            per_training_seed_scores_x100=values, independent_training_seeds=5)
        changes = [row['stochastic_minus_mean_score_points'] for row in rows]
        summaries[method]['stochastic_minus_mean'] = dict(mean_score_points=statistics.mean(changes),
                                                        training_seed_sample_sd=statistics.stdev(changes),
                                                        per_training_seed_changes=changes)
    gaps = {mode: summaries['f1'][mode]['mean_score_x100'] - summaries['props'][mode]['mean_score_x100']
            for mode in ('mean', 'stochastic')}
    output = dict(schema='chapter04-action-sampling-diagnostic-results-v1', status='COMPLETE_ALL_PRESPECIFIED_RESULTS',
                  plan_sha256=seal['plan_sha256'], methods=summaries, TRoC_minus_PROPS_score_points=gaps,
                  per_training_seed=per_seed, raw_sources=jobs,
                  verification=dict(case_records_recomputed=case_count, complete_paths=case_count,
                                    physical_environment_steps=steps, training_steps=0, new_independent_TEST_steps=0,
                                    evaluation_panels=90, max_aggregate_recalculation_error=max_recalc_error,
                                    max_relative_allocation_error=max_resource_error,
                                    same_DEV32_plan_hashes_across_all_models_and_repeats=True,
                                    protected_inputs_SHA_unchanged=len(protected)),
                  statistical_scope='Five training seeds per method; eight action repeats are repeated measurements, not extra independent training seeds. Fixed current DEV32, endpoint only; no new TEST or full learning curve.')
    dump(ROOT/'results/summary.json', output)
    fields = ['method', 'training_seed', 'mean_score_x100', 'stochastic_score_x100',
              'stochastic_repeat_sd_x100', 'stochastic_minus_mean_score_points']
    with (ROOT/'results/per_training_seed.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(per_seed)
    rows = []
    for method, label in [('f1', '本文TRoC'), ('props', 'PROPS')]:
        s = summaries[method]
        rows.append(f"   \"{label}\", \"{s['mean']['mean_score_x100']:.6f} ± {s['mean']['training_seed_sample_sd_x100']:.6f}\", \"{s['stochastic']['mean_score_x100']:.6f} ± {s['stochastic']['training_seed_sample_sd_x100']:.6f}\", \"{s['stochastic_minus_mean']['mean_score_points']:+.6f}\"")
    report = f'''第四章应急场景：随机动作与均值动作评价诊断
================================================

本次按用户要求实际重评当前正式终点模型，固定同一DEV32面板，只切换评价动作。没有重新训练、增加独立TEST或替换正式论文曲线。

.. csv-table:: 需求完成比例（百分数；±为五次独立训练的样本标准差）
   :header: "方法", "均值动作", "原生随机动作", "随机减均值（百分点）"

{chr(10).join(rows)}

TRoC减PROPS：均值动作 {gaps['mean']:+.6f} 个百分点，随机动作 {gaps['stochastic']:+.6f} 个百分点。

如何理解两种评价
------------------

旧测试沿用原生随机评价，每步从已训练高斯策略抽动作，温度为1；当前正式比较预先统一采用关闭探索后的部署动作，高斯策略取均值，原生确定性策略用原生动作。训练仍然随机。两者回答不同的执行问题，确定性评价并非天然更正确。随机动作改变实际资源配置及其后继状态，因而随机执行回报的平均值通常不等于均值动作的回报。

新旧任务的物理环境相同，但旧70.100373%还对应零输出头初始化、两训练种子、24576步以及不同开发情景。当前是双随机输出头、五训练种子和100608步。因此本诊断可隔离当前终点的执行方式效应，不能把历史70.10%到当前成绩的全部差异归因于随机评价。

本次均值重评对应原始最后开发点。图4.4(b)的79.860248%是PROPS最近五个开发点的平滑值；同一正式终点的原始开发均值是79.920679%，本次应对照后者。

完整设计与核验
----------------

当前v5的TRoC和PROPS各五个固定终点，原模型SHA先验核对；DEV master=202611043101，phase=dev，case0至31。每个模型均值评价一遍，随机评价八遍。抽样种子=91090000+1000×训练种子+重复序号；两方法使用相同数值种子，但实现与批量顺序不同，不声称逐动作标准噪声相同。F1使用独立torch.Generator，PROPS使用保存与恢复的全局torch RNG，直接调用原生随机策略；不缩放已学习标准差，也不额外加噪声。

统计先在每个训练种子内平均八次随机重复，再对五个训练种子等权汇总；八次动作重复不是八个新增训练种子。全部十模型、九十评价面板、2880完整过程、69120物理评价步均纳入，0学习步、0新独立TEST步。确定性逐案例重现原终点评价；模型张量、文件SHA、导入科学源、外生情景计划、全额投入与纯终奖均核验。逐案例数据、每次重复和每个训练种子结果完整保留，未按高低挑选。

本结果为已看过正式结果后的DEV终点诊断，不重建整条学习曲线，不作新的独立确认结论。现有终点以外的中间模型没有保存，本次不能用终点随机评价替代图中整条历史过程。

机器结果见results/summary.json，逐训练种子见results/per_training_seed.csv，逐案例见jobs/*/case_results.csv.gz，冻结方案见protocol/PLAN.json。历史理由来源为reviews/experiment_design_20261106/第四章_整体实验设计与训练计数.rst及旧学习率筛查的source/native_loop.py。
'''
    (ROOT/'随机与均值动作评价结果.rst').write_text(report)
    dump(ROOT/'results/computational_ledger.json', dict(training_steps=0, DEV_evaluation_steps=steps,
          new_independent_TEST_steps=0, complete_episodes=case_count, model_loads=10,
          model_files_created=0, optimization_updates=0, thesis_builds=0,
          episodes_by_method={method: 1440 for method in ('f1', 'props')},
          steps_by_method={method: 34560 for method in ('f1', 'props')}))
    print(json.dumps(dict(methods=summaries, gaps=gaps, verification=output['verification']), ensure_ascii=False))


if __name__ == '__main__':
    main()
