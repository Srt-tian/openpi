#!/usr/bin/env python3
"""One task, isolated plugin outputs; no silent skipping of unready modules."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request


def resolve(config, requested=None):
    if config.get('schema') != 'pi05-capability-bundle.v1':
        raise ValueError('unsupported bundle schema')
    modules = config['modules']
    names = [m['id'] for m in modules]
    if len(names) != len(set(names)):
        raise ValueError('duplicate module ID')
    selected = names if requested is None else requested
    if not selected or len(selected) != len(set(selected)) or set(selected)-set(names):
        raise ValueError('unknown, empty or duplicate selection')
    result = [m for m in modules if m['id'] in selected]
    for m in result:
        if m['status'] == 'ready' and m['trainer'] != 'demo_support_cfn_v1':
            raise ValueError('trainer not implemented: '+m['id'])
    return result


def stop_owned(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=20)


def run_logged(command, log, deadline, env):
    remaining = deadline-time.monotonic()
    if remaining <= 0:
        raise TimeoutError('bundle wall-time limit reached')
    with log.open('wb') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        try:
            code = child.wait(timeout=remaining)
            if code:
                raise RuntimeError('stage failed: '+log.name+' exit='+str(code))
        finally:
            stop_owned(child)


def verify_checkout(root, expected):
    head = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=all'],text=True)
    upstream = subprocess.check_output(['git','-C',str(root),'rev-parse','@{upstream}'],text=True).strip()
    origin = subprocess.check_output(['git','-C',str(root),'remote','get-url','origin'],text=True).strip()
    if head != expected or dirty or upstream != expected:
        raise ValueError('checkout must be clean and equal approved HEAD/upstream')
    if origin != 'https://github.com/Srt-tian/openpi.git':
        raise ValueError('checkout is not backed by the canonical remote')


def train_cfn(config, root, output, expected, env, deadline):
    python = sys.executable
    data, base = config['dataset'],config['base']
    service = config['service']
    cache = output/'cache'
    scripts = root/'scripts'
    service_cmd = [python,str(scripts/'serve_pi05_feature_probe.py'),
        '--base',base,'--base-sha256',config['base_sha256'],
        '--norm-sha256',config['norm_sha256'],'--expected-source-commit',expected,
        '--port',str(service['port'])]
    cache_cmd = [python,str(scripts/'build_pi05_demo_support_cache.py'),
        '--data',data,'--base',base,'--base-sha256',config['base_sha256'],
        '--norm-sha256',config['norm_sha256'],'--source-manifest',config['split_manifest'],
        '--output',str(cache),'--feature-endpoint','127.0.0.1',
        '--feature-port',str(service['port']),'--expected-service-source',expected,'--execute']
    process = None
    with (output/'feature_service.log').open('wb') as log:
        try:
            process = subprocess.Popen(service_cmd,stdout=log,stderr=subprocess.STDOUT,
                                       env=env,start_new_session=True)
            ready_deadline = min(deadline,time.monotonic()+service['startup_timeout_seconds'])
            while True:
                if process.poll() is not None:
                    raise RuntimeError('owned feature service exited before readiness')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:'+str(service['port'])+'/healthz',timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() >= ready_deadline:
                    raise TimeoutError('feature service readiness timed out')
                time.sleep(1)
            # Cache builder checks exact role/source/base/norm/schema metadata.
            run_logged(cache_cmd,output/'feature_cache.log',deadline,env)
        finally:
            stop_owned(process)
    # The frozen JAX service has exited before the PyTorch fit starts.
    train_cmd = [python,str(scripts/'train_demo_support_cfn.py'),
        '--cache',str(cache),'--output',str(output/'checkpoints'),
        '--expected-source',expected,'--base-sha256',config['base_sha256'],
        '--norm-sha256',config['norm_sha256'],
        '--dataset-manifest-sha256',config['split_manifest_sha256'],
        '--device','cuda:0','--updates','2000','--batch-size','256',
        '--seed','42','--lr','1e-4','--weight-decay','1e-4','--execute']
    run_logged(train_cmd,output/'training.log',deadline,env)
    trained = json.loads((output/'checkpoints'/'manifest.json').read_text())
    if trained['selection'] != 'fixed_final_step_2000_not_best_of_validation' or trained['metrics'][-1]['step'] != 2000:
        raise ValueError('final checkpoint receipt mismatch')
    return trained


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--modules',nargs='+',help='default: every listed trainable module, including blocked entries')
    parser.add_argument('--expected-source')
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    selected = resolve(config,args.modules)
    blocked = [{'id':m['id'],'reason':m['reason']} for m in selected if m['status'] != 'ready']
    receipt = {'schema':'pi05-capability-bundle-receipt.v1','status':'blocked' if blocked else 'preflight_only',
        'selected':[m['id'] for m in selected],
        'excluded':[m['id'] for m in config['modules'] if m not in selected],
        'blocked':blocked,'outputs':{},'image_reference':config['image_reference']}
    print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
    if blocked:
        return 2
    if not args.execute:
        return 0
    if os.environ.get('PI05_BUNDLE_AUTHORIZATION') != 'CONFIRMED' or not args.expected_source:
        raise ValueError('execution needs explicit confirmed configuration and expected source')
    root = Path(__file__).resolve().parents[1]
    verify_checkout(root,args.expected_source)
    for path in (config['dataset'],config['base'],config['split_manifest']):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    import hashlib
    if hashlib.sha256(Path(config['split_manifest']).read_bytes()).hexdigest() != config['split_manifest_sha256']:
        raise ValueError('split manifest identity mismatch')
    # Refuse an existing service; never borrow or stop another job's endpoint.
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',config['service']['port']))
    output = Path(config['output'])
    output.mkdir(parents=True,exist_ok=False)
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES']='0'
    env['JAX_PLATFORMS']='cuda'
    env['PYTHONPATH']=os.pathsep.join([str(root),str(root/'src'),str(root/'scripts'),str(root/'packages/openpi-client/src'),env.get('PYTHONPATH','')])
    receipt.update(status='running',source_commit=args.expected_source,config=config)
    status = output/'bundle_status.json'
    deadline = time.monotonic()+config['wall_time_limit_seconds']
    status.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    try:
        for module in selected:
            destination = output/module['id']
            destination.mkdir()
            receipt['outputs'][module['id']]={'status':'running','directory':str(destination)}
            status.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
            trained = train_cfn(config,root,destination,args.expected_source,env,deadline)
            receipt['outputs'][module['id']].update(status='complete',final_checkpoint=trained['metrics'][-1])
        receipt['status']='complete'
    except BaseException as exc:
        receipt.update(status='failed',error_type=type(exc).__name__)
        raise
    finally:
        status.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
