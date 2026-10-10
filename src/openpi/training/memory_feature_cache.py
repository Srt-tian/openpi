"""Hash-bound, non-pickle, sharded memory training cache and causal row reader."""
import hashlib
import json
from pathlib import Path
import numpy as np
from openpi.training import native_memory_features as feature


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def bound_path(root,relative):
    path=(root/relative).resolve()
    if Path(relative).is_absolute() or root.resolve() not in path.parents:
        raise ValueError('cache file escapes root')
    return path


def feature_arrays(values,count):
    expected={'hidden':(count,4,10,1024),'base_velocity':(count,4,10,32),'target_velocity7':(count,4,10,7)}
    if set(values)!=set(expected):raise ValueError('feature array fields mismatch')
    for key,shape in expected.items():
        if values[key].dtype!=np.float32 or values[key].shape!=shape or not np.isfinite(values[key]).all():
            raise ValueError('invalid '+key)


class CacheWriter:
    def __init__(self,root,*,source,base_sha,norm_sha,split_sha,dataset_meta,shard_size=64):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False)
        if type(shard_size) is not int or shard_size<1:raise ValueError('invalid shard size')
        self.shard_size=shard_size;self.buffers={s:[] for s in ('train','val')}
        self.manifest={'schema':'pi05-memory-cache.v1','status':'building','source_commit':source,
            'base_sha256':base_sha,'norm_sha256':norm_sha,'split_manifest_sha256':split_sha,
            'dataset_meta':dataset_meta,'feature_schema':feature.schema(base_sha,norm_sha),
            'episodes':{},'shards':{},'rows':{'train':[],'val':[]}}
        self.keys=set();self.save()
    def save(self):
        (self.root/'manifest.json').write_text(json.dumps(self.manifest,sort_keys=True,indent=2)+'\n')
    def episode(self,split,episode,task,transitions):
        if split not in self.buffers or type(episode) is not int or episode<0 or type(task) is not int:
            raise ValueError('invalid episode identity')
        key=f'{split}:{episode}'
        if key in self.manifest['episodes']:raise ValueError('duplicate episode')
        value=np.asarray(transitions)
        if value.dtype!=np.float32 or value.ndim!=2 or value.shape[1]!=23 or not np.isfinite(value).all():
            raise ValueError('invalid episode transitions')
        path=f'episode_{split}_{episode}.npz';np.savez(self.root/path,transitions=value)
        self.manifest['episodes'][key]={'file':path,'sha256':sha(self.root/path),'frames':len(value)+1,'task':task}
    def append(self,row,response):
        split=row['split'];key=row['key']
        if split not in self.buffers or key in self.keys:raise ValueError('invalid/duplicate row')
        if not np.array_equal(np.asarray(response['time']),np.asarray(feature.TIMES,np.float32)):
            raise ValueError('wrong early flow times')
        arrays={k:np.asarray(response[k])[None] for k in ('hidden','base_velocity','target_velocity7')}
        feature_arrays(arrays,1)
        self.keys.add(key);self.buffers[split].append((dict(row),response))
        if len(self.buffers[split])>=self.shard_size:self.flush(split)
    def flush(self,split):
        pending=self.buffers[split]
        if not pending:return
        path=f'features_{split}_{len(self.manifest["shards"]):05d}.npz'
        arrays={k:np.stack([response[k] for _,response in pending]) for k in ('hidden','base_velocity','target_velocity7')}
        feature_arrays(arrays,len(pending));np.savez(self.root/path,**arrays)
        self.manifest['shards'][path]={'sha256':sha(self.root/path),'count':len(pending),'split':split}
        for offset,(row,_) in enumerate(pending):
            self.manifest['rows'][split].append({k:row[k] for k in ('key','episode','frame','task','split')}|{'shard':path,'offset':offset})
        self.buffers[split]=[];self.save()
    def finish(self,expected_counts):
        for split in self.buffers:self.flush(split)
        if any(len(self.manifest['rows'][s])!=expected_counts[s] for s in self.buffers):
            raise ValueError('cache coverage differs from selected plan')
        self.manifest['status']='complete';self.save()
        # Independently read back the complete receipt before declaring success.
        try:CacheReader(self.root)
        except BaseException as error:
            self.failed(error);raise
    def failed(self,error):
        self.manifest.update(status='failed',error_type=type(error).__name__);self.save()


class CacheReader:
    def __init__(self,root):
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text())
        m=self.manifest
        if m.get('schema')!='pi05-memory-cache.v1' or m.get('status')!='complete':
            raise ValueError('memory cache is incomplete')
        if m['feature_schema']!=feature.schema(m['base_sha256'],m['norm_sha256']):
            raise ValueError('memory cache feature schema mismatch')
        self.histories={};self.features={};self.rows=m['rows'];episode_sets={s:set() for s in ('train','val')}
        for key,record in m['episodes'].items():
            split,episode=key.split(':');episode_sets[split].add(int(episode))
            path=bound_path(self.root,record['file'])
            if sha(path)!=record['sha256']:raise ValueError('history hash mismatch')
            with np.load(path,allow_pickle=False) as z:
                x=z['transitions']
                if set(z.files)!={'transitions'} or x.dtype!=np.float32 or x.shape!=(record['frames']-1,23) or not np.isfinite(x).all():
                    raise ValueError('invalid episode history array')
                self.histories[key]=x
        if episode_sets['train']&episode_sets['val']:raise ValueError('episode split overlap')
        for name,record in m['shards'].items():
            path=bound_path(self.root,name)
            if sha(path)!=record['sha256']:raise ValueError('feature hash mismatch')
            with np.load(path,allow_pickle=False) as z:
                arrays={k:z[k] for k in z.files};feature_arrays(arrays,record['count'])
                self.features[name]=arrays
        keys=set();positions=set()
        for split in ('train','val'):
            for row in self.rows[split]:
                key=f'{split}:{row["episode"]}';record=m['episodes'][key];shard=m['shards'][row['shard']]
                if row['split']!=split or shard['split']!=split or row['task']!=record['task'] or not 0<=row['frame']<record['frames']:
                    raise ValueError('row/history split, task or frame mismatch')
                if not 0<=row['offset']<shard['count']:raise ValueError('row offset outside shard')
                if row['key']!=f'long/{split}/episode:{row["episode"]}/frame:{row["frame"]}':
                    raise ValueError('logical training key mismatch')
                position=(row['shard'],row['offset'])
                if row['key'] in keys or position in positions:raise ValueError('duplicate cache row')
                keys.add(row['key']);positions.add(position)
            if len({r['task'] for r in self.rows[split]})!=10:raise ValueError('must cover ten tasks per split')
        if len(positions)!=sum(x['count'] for x in m['shards'].values()):raise ValueError('unreferenced cache rows')
        if {r['task'] for r in self.rows['train']}!={r['task'] for r in self.rows['val']}:
            raise ValueError('train/validation task coverage mismatch')
    def item(self,split,index,time_index):
        if split not in ('train','val') or type(time_index) is not int or not 0<=time_index<4:
            raise ValueError('invalid split or early time index')
        row=self.rows[split][index];arrays=self.features[row['shard']];frame=row['frame']
        history=self.histories[f'{split}:{row["episode"]}'][max(0,frame-520):frame].copy()
        result={k:arrays[k][row['offset'],time_index].copy() for k in arrays}
        return result|{'history':history,'time':np.float32(feature.TIMES[time_index])}
