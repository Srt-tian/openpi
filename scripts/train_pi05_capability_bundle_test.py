import json
import subprocess
import sys
import pytest
import train_pi05_capability_bundle as bundle


def config():
    return {'schema':'pi05-capability-bundle.v1','image_reference':'reference',
        'modules':[{'id':'cfn','status':'ready','trainer':'demo_support_cfn_v1'},
                   {'id':'memory','status':'blocked','reason':'missing trainer'}]}


def test_default_all_rejects_missing_modules_without_execution(tmp_path,monkeypatch,capsys):
    path=tmp_path/'plan.json';path.write_text(json.dumps(config()))
    monkeypatch.setattr(bundle,'train_cfn',lambda *a:pytest.fail('must not train'))
    assert bundle.main(['--config',str(path),'--execute'])==2
    receipt=json.loads(capsys.readouterr().out)
    assert receipt['selected']==['cfn','memory'] and receipt['status']=='blocked'
    assert receipt['outputs']=={} and receipt['excluded']==[]


def test_explicit_subset_is_reported_and_dryrun_has_no_outputs(tmp_path,capsys):
    path=tmp_path/'plan.json';path.write_text(json.dumps(config()))
    assert bundle.main(['--config',str(path),'--modules','cfn'])==0
    receipt=json.loads(capsys.readouterr().out)
    assert receipt['excluded']==['memory'] and receipt['status']=='preflight_only'
    assert set(p.name for p in tmp_path.iterdir())=={'plan.json'}


def test_execution_requires_confirmation(tmp_path,monkeypatch):
    path=tmp_path/'plan.json';path.write_text(json.dumps(config()))
    monkeypatch.delenv('PI05_BUNDLE_AUTHORIZATION',raising=False)
    with pytest.raises(ValueError,match='confirmed'):
        bundle.main(['--config',str(path),'--modules','cfn','--execute'])


@pytest.mark.parametrize('selection',[[],['bogus'],['cfn','cfn']])
def test_invalid_selection(selection):
    with pytest.raises(ValueError):bundle.resolve(config(),selection)


def test_unknown_trainer_cannot_be_marked_ready():
    value=config();value['modules'][0]['trainer']='pretend_memory'
    with pytest.raises(ValueError,match='not implemented'):bundle.resolve(value)


def test_failed_stage_is_not_success(tmp_path):
    import time
    import os
    with pytest.raises(RuntimeError,match='exit=7'):
        bundle.run_logged([sys.executable,'-c','raise SystemExit(7)'],tmp_path/'failure.log',time.monotonic()+5,dict(os.environ))
