import json
from pathlib import Path
import pytest
import run_pi05_memory_task as task


def config():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/pi05_memory_task.json').read_text())


def test_no_execution_or_files_without_specific_confirmation(tmp_path,monkeypatch,capsys):
    plan=tmp_path/'config.json';plan.write_text(json.dumps(config()))
    monkeypatch.setattr(task,'execute',lambda *a:pytest.fail('must not run'))
    task.main(['--config',str(plan)])
    assert json.loads(capsys.readouterr().out)['gpu_work_started'] is False
    monkeypatch.delenv('PI05_MEMORY_AUTHORIZATION',raising=False)
    with pytest.raises(ValueError,match='confirmed'):task.main(['--config',str(plan),'--execute'])


def test_job_commands_have_exact_identities_and_separate_outputs(tmp_path):
    commands=task.stages(config(),Path('/code'),tmp_path,'c'*40)
    assert '--expected-source-commit' in commands['service'] and 'c'*40 in commands['service']
    assert str(tmp_path/'cache') in commands['cache'] and str(tmp_path/'checkpoints') in commands['fit']
    assert '--execute' in commands['head_smoke']


def test_failed_owned_service_never_starts_cache():
    class Dead:
        def poll(self):return 3
    with pytest.raises(RuntimeError,match='owned'):task.wait_service(Dead(),18672,10**12)


def test_training_cannot_start_until_owned_service_exits(tmp_path,monkeypatch):
    order=[]
    class Service:
        pid=1234
        def __init__(self,*a,**k):self.stopped=False
        def poll(self):return 0 if self.stopped else None
    def stop(process):
        if process is not None:process.stopped=True;order.append('service_stop')
    def run(command,*a):
        script=Path(command[1]).name;order.append(script)
        if script=='train_pi05_execution_memory.py':
            assert 'service_stop' in order
            raise RuntimeError('fixture fit failed')
    monkeypatch.setattr(task.subprocess,'Popen',Service)
    monkeypatch.setattr(task,'run_logged',run)
    monkeypatch.setattr(task,'stop_owned',stop)
    monkeypatch.setattr(task,'wait_service',lambda *a:None)
    receipt={'status':'running'}
    with pytest.raises(RuntimeError,match='fixture'):task.execute(config(),Path('/code'),tmp_path,'c'*40,{},receipt)
    assert json.loads((tmp_path/'task_status.json').read_text())['status']=='failed'
    assert order.index('service_stop')<order.index('train_pi05_execution_memory.py')
