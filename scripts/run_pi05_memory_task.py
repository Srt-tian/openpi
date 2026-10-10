#!/usr/bin/env python3
"""Single owned GPU lane: synthetic smoke, feature cache, then memory fit."""
import argparse,json,os,socket,subprocess,sys,time,urllib.request
from pathlib import Path
from train_pi05_capability_bundle import run_logged,stop_owned,verify_checkout


def validate(config):
    if config.get('schema')!='pi05-memory-task.v1':raise ValueError('memory task schema mismatch')
    expected={'updates':3000,'batch':16,'seed':42,'lr':1e-4,'weight_decay':1e-4,'clip':1,'checkpoint_every':500}
    if config['training']!=expected:raise ValueError('task and trainer recipe differ')
    if config['port']!=18672 or config['wall_seconds']!=14400:raise ValueError('pilot budget/port differs')
    if not config['image'].endswith('@sha256:5d5fbba8f0cbe5ea64bee17f655085f485c197f8966c387464c46eba46d51f7c'):
        raise ValueError('image reference is not the proposed immutable image')


def stages(config,root,output,source):
    scripts=root/'scripts';python=sys.executable
    identity=['--base',config['base'],'--base-sha256',config['base_sha256'],
        '--norm-sha256',config['norm_sha256'],'--expected-source-commit',source]
    return {
        'head_smoke':[python,str(scripts/'smoke_pi05_memory_gpu.py'),'--output',str(output/'head_smoke.json'),'--execute'],
        'native_smoke':[python,str(scripts/'probe_pi05_native_features.py'),*identity],
        'service':[python,str(scripts/'serve_pi05_memory_features.py'),*identity,'--port',str(config['port'])],
        'cache':[python,str(scripts/'build_pi05_memory_cache.py'),'--data',config['data'],'--norm',config['norm'],
            '--split-manifest',config['split_manifest'],'--output',str(output/'cache'),
            '--base-sha256',config['base_sha256'],'--norm-sha256',config['norm_sha256'],
            '--expected-source',source,'--feature-port',str(config['port']),'--execute'],
        'fit':[python,str(scripts/'train_pi05_execution_memory.py'),'--cache',str(output/'cache'),
            '--output',str(output/'checkpoints'),'--expected-source',source,
            '--base-sha256',config['base_sha256'],'--norm-sha256',config['norm_sha256'],
            '--split-sha256',config['split_sha256'],'--execute']}


def wait_service(process,port,deadline):
    limit=min(deadline,time.monotonic()+300)
    while True:
        if process.poll() is not None:raise RuntimeError('owned memory feature service exited')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz',timeout=2) as response:
                if response.status==200:
                    if process.poll() is not None:raise RuntimeError('owned service died during readiness')
                    return
        except OSError:pass
        if time.monotonic()>=limit:raise TimeoutError('memory feature service readiness timed out')
        time.sleep(1)


def execute(config,root,output,source,env,receipt):
    commands=stages(config,root,output,source);deadline=time.monotonic()+config['wall_seconds']
    status=output/'task_status.json'
    def save():status.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    def stage(name):
        receipt['phase']=name;save();run_logged(commands[name],output/(name+'.log'),deadline,env)
    service=None
    try:
        stage('head_smoke');stage('native_smoke')
        with (output/'service.log').open('wb') as stream:
            try:
                service=subprocess.Popen(commands['service'],stdout=stream,stderr=subprocess.STDOUT,
                                         start_new_session=True,env=env)
                receipt.update(phase='service_start',owned_service_pid=service.pid);save()
                wait_service(service,config['port'],deadline)
                stage('cache')
            finally:stop_owned(service)
        if service.poll() is None:raise RuntimeError('service must exit before fit')
        receipt['feature_service_released']=True;save();stage('fit')
        trained=json.loads((output/'checkpoints'/'manifest.json').read_text())
        if trained['status']!='complete' or trained['metrics'][-1]['step']!=3000 or trained['source_commit']!=source:
            raise ValueError('training final receipt mismatch')
        from openpi.training.memory_feature_cache import sha
        final=trained['metrics'][-1]
        if sha(output/'checkpoints'/final['checkpoint'])!=final['sha256']:raise ValueError('final checkpoint digest mismatch')
        receipt.update(status='complete',phase='finished',final_checkpoint=final)
    except BaseException as error:
        receipt.update(status='failed',error_type=type(error).__name__);raise
    finally:stop_owned(service);save()


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--expected-source');p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
    config=json.loads(a.config.read_text());validate(config)
    if not a.execute:
        print(json.dumps({'status':'preflight_only','training':config['training'],'image':config['image'],
            'output':config['output'],'gpu_work_started':False},sort_keys=True));return
    if os.environ.get('PI05_MEMORY_AUTHORIZATION')!='CONFIRMED' or not a.expected_source:
        raise ValueError('explicit confirmed memory configuration and source required')
    root=Path(__file__).resolve().parents[1];verify_checkout(root,a.expected_source)
    from openpi.training.memory_feature_cache import sha
    for key in ('data','base','norm','split_manifest'):
        if not Path(config[key]).exists():raise FileNotFoundError(config[key])
    if sha(config['norm'])!=config['norm_sha256'] or sha(config['split_manifest'])!=config['split_sha256']:
        raise ValueError('norm or demonstration split identity mismatch')
    with socket.socket() as probe:probe.bind(('127.0.0.1',config['port']))
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='0',JAX_PLATFORMS='cuda',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    env['PYTHONPATH']=os.pathsep.join([str(root),str(root/'src'),str(root/'scripts'),str(root/'packages/openpi-client/src'),env.get('PYTHONPATH','')])
    receipt={'schema':'pi05-memory-task-receipt.v1','status':'running','source_commit':a.expected_source,
        'config':config,'phase':'preflight_complete','benchmarks_started':False}
    execute(config,root,output,a.expected_source,env,receipt)


if __name__=='__main__':main()
