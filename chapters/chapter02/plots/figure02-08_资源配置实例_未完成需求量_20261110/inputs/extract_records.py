"""Extract six decision snapshots from the retained fixed-policy test traces."""
from pathlib import Path
import csv, hashlib, json
import numpy as np

BASE=Path(__file__).resolve().parent
PARENT=Path('/hy-tmp/resource_allocation_results/thesis_hex_experiment_adoption_20261108_194130/assets/c2')
PERIODS={'radio':[1,16,32],'v5':[1,12,24]}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    original=json.loads((PARENT/'data/original_source.json').read_text())
    with (PARENT/'data.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;prior=list(reader)
    old={(r['domain'],int(r['period']),int(r['object_id'])):r for r in prior}
    rows=[]
    for domain in ['radio','v5']:
        source=original['case_source'][domain]
        path=Path(source['path']);assert sha(path)==source['sha256']
        with np.load(path,allow_pickle=False) as z:
            for period in PERIODS[domain]:
                index=period-1;count=100 if domain=='radio' else 120
                for obj in range(1,count+1):
                    row={k:'' for k in fields}
                    row.update(domain=domain,training_seed='0',test_path_index='0',period=str(period),object_id=str(obj),action_native=str(float(z['path_0_actions'][index,obj-1])))
                    if domain=='radio':
                        real=z['path_0_channel_real'][index,obj-1];imag=z['path_0_channel_imag'][index,obj-1]
                        row.update(channel_real=str(float(real)),channel_imag=str(float(imag)),channel_power=str(float(real**2+imag**2)),confirmed_packets=str(float(z['path_0_confirmed_packets'][index,obj-1])))
                    else:
                        row.update(remaining_demand=str(float(z['path_0_remaining_demand'][index,obj-1])),total_demand_so_far=str(float(z['path_0_total_demand_so_far'][index,obj-1])))
                    key=(domain,period,obj)
                    if key in old:
                        for k,v in old[key].items():
                            assert (v==row[k]) or (v and row[k] and float(v)==float(row[k])),(key,k,v,row[k])
                        row=dict(old[key])
                    rows.append(row)
    assert len(rows)==660
    assert all(next(r for r in rows if (r['domain'],int(r['period']),int(r['object_id']))==key)==v for key,v in old.items())
    with (BASE/'data.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    provenance={'schema':'chapter2-retained-fixed-trace-three-timepoints-v1','parent_directory':str(PARENT),'parent_data_sha256':sha(PARENT/'data.csv'),
        'case_source':original['case_source'],'periods':PERIODS,'selected_path_index':0,'training_seed':0,'records':660,
        'midpoint_records':220,'midpoint_parent_rows_exactly_preserved':True,'data_sha256':sha(BASE/'data.csv'),
        'model_loads':0,'environment_steps':0,'new_policy_inference':0,'training_updates':0,
        'state_timing':'decision-before-action state paired with that period actual action, as retained in the original trace'}
    (BASE/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'PASS','records':660,'midpoint_records':220,'data_sha256':provenance['data_sha256']}))

if __name__=='__main__':main()
