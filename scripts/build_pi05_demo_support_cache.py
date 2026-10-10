#!/usr/bin/env python3
"""Build a bound Long-suite demo feature cache; dry-run is the default."""
import argparse, hashlib, json, subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def frame_plan(dataset, split):
    rows=[];start=0
    for end,ep in zip(dataset._ends.tolist(),dataset.episodes,strict=True):
        offsets=sorted(set(range(0,ep.length,5))|{ep.length-1})
        rows += [{"key":f"long/{split}/episode:{ep.index}/frame:{f}","index":start+f,
                  "episode":ep.index,"frame":f,"task":ep.task,"split":split} for f in offsets]
        start=int(end)
    return rows
def validate_plan(train,val):
    if {x['episode'] for x in train}&{x['episode'] for x in val}: raise ValueError('episode split overlap')
    keys=[x['key'] for x in train+val]
    if len(keys)!=len(set(keys)) or any(x['frame']<0 for x in train+val): raise ValueError('invalid frame keys')
    return {"train":len(train),"val":len(val),"tasks":sorted({x['task'] for x in train})}
def validate_service_metadata(actual,expected):
    if actual!=expected:raise ValueError('feature service identity mismatch')
    return True
def prepare_output(path):
    path.mkdir(parents=True);status=path/'status.json';status.write_text(json.dumps({'status':'extracting'})+'\n');return status
def parse(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--base-sha256',required=True);p.add_argument('--norm-sha256',required=True);p.add_argument('--source-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--feature-endpoint');p.add_argument('--feature-port',type=int);p.add_argument('--expected-service-source');return p.parse_args(argv)
def main(argv=None):
    a=parse(argv);root=Path(__file__).resolve().parents[1]
    from openpi.training.plugin_data import build_suite_datasets
    train,val,dm=build_suite_datasets(a.data,horizon=10,seed=42,holdout_per_task=2)
    tr,va=train['libero_10'],val['libero_10'];plans={"train":frame_plan(tr,'train'),"val":frame_plan(va,'val')};coverage=validate_plan(plans['train'],plans['val'])
    approved=json.loads(a.source_manifest.read_text())['dataset_manifest']['suites']['libero_10']
    if set(approved['train_episode_indices'])!={e.index for e in tr.episodes} or set(approved['val_episode_indices'])!={e.index for e in va.episodes}:
        raise ValueError('V2 Long split membership mismatch')
    if a.output.exists(): raise FileExistsError(a.output)
    pre={"status":"ready" if a.execute else "dry_run_only","coverage":coverage,"episodes":{"train":len(tr.episodes),"val":len(va.episodes)}}
    if not a.execute: print(json.dumps(pre,sort_keys=True));return
    if not a.feature_endpoint or not a.feature_port or not a.expected_service_source:raise ValueError('execute requires verified remote feature endpoint')
    status_path=prepare_output(a.output)
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from openpi.training import native_feature_probe as fp
    from serve_pi05_feature_probe import ROLE
    client=WebsocketClientPolicy(a.feature_endpoint,a.feature_port);meta=client.get_server_metadata()
    expected={'role':ROLE,'source_commit':a.expected_service_source,'base_sha256':a.base_sha256,'norm_sha256':a.norm_sha256,
      'feature_schema':fp.schema_record(base_sha256=a.base_sha256,norm_sha256=a.norm_sha256)}
    validate_service_metadata(meta,expected)
    source=a.expected_service_source;base_hash=a.base_sha256;norm_hash=a.norm_sha256;arrays={}
    try:
      for split,ds in (("train",tr),("val",va)):
        feats=[]
        for row in plans[split]:
            sample=ds[row['index']];physical=sample.pop('actions')
            noise=fp.common_noise(fp.feature_probe_seed(int.from_bytes(hashlib.sha256(row['key'].encode()).digest()[:8],'big'),0))
            response=client.infer({**sample,'chunks7':physical[None],'noise32':noise});feature=np.asarray(response.get('feature'),np.float32)
            if feature.shape!=(1,1024) or not np.isfinite(feature).all():raise ValueError('invalid remote feature response')
            feats.append(feature[0])
        arrays[split]=np.stack(feats)
    except Exception as exc:
      status_path.write_text(json.dumps({'status':'failed','error_type':type(exc).__name__})+'\n');raise
    np.savez(a.output/'features.npz',train=arrays['train'],val=arrays['val'],
      train_keys=np.asarray([x['key'] for x in plans['train']]),val_keys=np.asarray([x['key'] for x in plans['val']]),
      train_tasks=np.asarray([x['task'] for x in plans['train']],np.int32),val_tasks=np.asarray([x['task'] for x in plans['val']],np.int32))
    cache_sha=sha(a.output/'features.npz');manifest={"schema":"demo_support_cache.v1","source_commit":source,"base_sha256":base_hash,
      "norm_sha256":norm_hash,"dataset_manifest_sha256":sha(a.source_manifest),"dataset_meta":{"info_sha256":sha(a.data/'meta/info.json'),
      "tasks_sha256":sha(a.data/'meta/tasks.parquet')},"feature_schema":fp.schema_record(base_sha256=base_hash,norm_sha256=norm_hash),
      "selection":"every_5_frames_plus_last","parent_target":"H10_official_repeat_last","coverage":coverage,"features_sha256":cache_sha,
      "arrays":{"train":list(arrays['train'].shape),"val":list(arrays['val'].shape)},"frame_keys":plans}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    status_path.write_text(json.dumps({'status':'complete','features_sha256':cache_sha})+'\n')
if __name__=='__main__':main()
