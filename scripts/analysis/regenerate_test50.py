"""Regenerate descriptive TEST50 summaries from the published per-state CSVs.

No policy inference, training, checkpoint selection, or hypothesis testing.
The two policies share 50 states; they are not 100 independent initial states.
"""
from pathlib import Path
import argparse,csv,json
from collections import Counter
import numpy as np

def regenerate(source: Path, output: Path):
    output.mkdir(parents=True,exist_ok=False)
    policies={name:list(csv.DictReader((source/f'{name}_per_state.csv').open())) for name in ('act300k','dp480k')}
    keys=('collection_order','planned_state_id','inference_seed','cell_row','cell_col','object_x','object_y')
    for key in keys:
        assert [r[key] for r in policies['act300k']]==[r[key] for r in policies['dp480k']],key
    rows=[];regions=[];stages=[];cells=[]
    for name,data in policies.items():
        assert len(data)==50 and len({r['planned_state_id'] for r in data})==50
        success=lambda r:r['success']=='True'
        n=sum(map(success,data));acq=sum(r['acquisition_proxy']=='True' for r in data)
        assert n=={'act300k':27,'dp480k':43}[name]
        rows.append(dict(policy=name,n=50,success=n,success_percent=100*n/50,acquired=acq,success_after_acquisition=sum(success(r) and r['acquisition_proxy']=='True' for r in data)))
        for region in ('LOWER','MID','UPPER'):
            subset=[r for r in data if r['test_region']==region];k=sum(map(success,subset))
            regions.append(dict(policy=name,region=region,n=len(subset),success=k,success_percent=100*k/len(subset)))
        counter=Counter(r['failure_stage'] for r in data)
        stages.extend(dict(policy=name,stage=s,count=counter[s]) for s in 'ABCDEF')
        for row in range(5):
            for col in range(5):
                subset=[r for r in data if int(r['cell_row'])==row and int(r['cell_col'])==col]
                assert len(subset)==2
                cells.append(dict(policy=name,row=row,col=col,n=2,success=sum(map(success,subset))))
    for name,data in [('results',rows),('regions',regions),('stages',stages),('cells',cells)]:
        with (output/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    labels=['ACT 300k','Diffusion Policy 480k'];colors=['#586dc4','#339985']
    axes[0].bar(labels,[r['success_percent'] for r in rows],color=colors)
    axes[0].set(ylim=(0,100),ylabel='Success (%)',title='Same frozen TEST50')
    for i,r in enumerate(rows):axes[0].text(i,r['success_percent']+2,f"{r['success']}/50 = {r['success_percent']:.0f}%",ha='center')
    for i,name in enumerate(policies):
        vals=[r['success_percent'] for r in regions if r['policy']==name]
        axes[1].bar(np.arange(3)+(i-.5)*.35,vals,width=.35,color=colors[i],label=labels[i])
        vals=[r['count'] for r in stages if r['policy']==name]
        axes[2].bar(np.arange(6)+(i-.5)*.35,vals,width=.35,color=colors[i],label=labels[i])
    axes[1].set(xticks=range(3),xticklabels=['Lower (20)','Mid (10)','Upper (20)'],ylim=(0,110),title='Spatial groups',ylabel='Success (%)')
    axes[2].set(xticks=range(6),xticklabels=list('ABCDEF'),title='Furthest observed stage (F = success)',ylabel='Episodes');axes[2].legend()
    fig.savefig(output/'comparison.png',dpi=180);plt.close(fig)
    report=dict(results=rows,paired_states=50,rollouts_executed=0,source='Published per-state summaries; full trajectory stage reclassification requires external action logs')
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[2]/'results/square_test50');p.add_argument('--output',required=True,type=Path);a=p.parse_args();print(json.dumps(regenerate(a.source,a.output),indent=2))
