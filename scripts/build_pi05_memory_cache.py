#!/usr/bin/env python3
"""Build native Long early-flow features and deduplicated causal histories."""
import argparse,json,os
from pathlib import Path
import numpy as np
from openpi.training import native_memory_features as feature
from openpi.training.memory_feature_cache import CacheWriter,sha
from build_pi05_demo_support_cache import frame_plan,validate_plan


def main(argv=None):
    p=argparse.ArgumentParser()
    for name in ('data','norm','split-manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('base-sha256','norm-sha256','expected-source'):p.add_argument('--'+name,required=True)
    p.add_argument('--feature-port',type=int);p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
    from openpi.training.plugin_data import build_suite_datasets
    from openpi.shared import normalize
    if sha(a.norm)!=a.norm_sha256:raise ValueError('normalization identity mismatch')
    train,val,_=build_suite_datasets(a.data,horizon=10,seed=42,holdout_per_task=2)
    datasets={'train':train['libero_10'],'val':val['libero_10']}
    plans={s:frame_plan(ds,s) for s,ds in datasets.items()};coverage=validate_plan(plans['train'],plans['val'])
    approved=json.loads(a.split_manifest.read_text())['dataset_manifest']['suites']['libero_10']
    for split in datasets:
        if {ep.index for ep in datasets[split].episodes}!=set(approved[f'{split}_episode_indices']):
            raise ValueError('approved demonstration split membership mismatch')
    if a.output.exists():raise FileExistsError(a.output)
    print(json.dumps({'mode':'execute' if a.execute else 'preflight_only','coverage':coverage,
        'four_time_feature_payload_bytes':(coverage['train']+coverage['val'])*4*10*1024*4},sort_keys=True),flush=True)
    if not a.execute:return
    if os.environ.get('PI05_MEMORY_AUTHORIZATION')!='CONFIRMED':
        raise ValueError('specific memory execution confirmation required')
    if not a.feature_port:raise ValueError('verified loopback feature service required')
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from serve_pi05_memory_features import ROLE
    client=WebsocketClientPolicy('127.0.0.1',a.feature_port)
    expected={'role':ROLE,'source_commit':a.expected_source,'feature_schema':feature.schema(a.base_sha256,a.norm_sha256)}
    if client.get_server_metadata()!=expected:raise ValueError('memory feature service identity mismatch')
    stats=normalize.load(a.norm.parent)
    writer=CacheWriter(a.output,source=a.expected_source,base_sha=a.base_sha256,norm_sha=a.norm_sha256,
        split_sha=sha(a.split_manifest),dataset_meta={'info_sha256':sha(a.data/'meta/info.json'),'tasks_sha256':sha(a.data/'meta/tasks.parquet')})
    try:
        for split,ds in datasets.items():
            for ep in ds.episodes:
                table=ds._load_episode(ep)
                pre=feature.quantile(table.state[:-1],stats['state'],8)
                post=feature.quantile(table.state[1:],stats['state'],8)
                actions=feature.quantile(table.action[:-1],stats['actions'],7)
                writer.episode(split,ep.index,ep.task,np.concatenate([pre,actions,post-pre],axis=-1).astype(np.float32))
            for row in plans[split]:
                sample=ds[row['index']];actions=sample.pop('actions')
                response=client.infer({**sample,'actions7':actions,'noise32':feature.cache_noise(row['key'])})
                writer.append(row,response)
        writer.finish({s:len(rows) for s,rows in plans.items()})
    except BaseException as error:
        writer.failed(error);raise


if __name__=='__main__':main()
