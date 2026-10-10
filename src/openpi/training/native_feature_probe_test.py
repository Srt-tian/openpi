from types import SimpleNamespace
import numpy as np
import pytest
import jax.numpy as jnp

from openpi.training import native_feature_probe as probe
import probe_pi05_native_features as cli


class FakeModel:
    def flow_features(self, observation, x_t, time):
        del observation,time
        # Position-specific 1024D final hidden makes first5-vs-all10 observable.
        pos=jnp.arange(10,dtype=jnp.float32)[None,:,None]
        return jnp.zeros_like(x_t),jnp.broadcast_to(pos,(x_t.shape[0],10,1024))+x_t[...,:1]


def stats(): return SimpleNamespace(q01=np.zeros(7,np.float32),q99=np.ones(7,np.float32))


def test_first_five_pooling_and_teacher_runtime_identity():
    chunks=np.zeros((2,10,7),np.float32);noise=np.zeros((10,32),np.float32)
    got=probe.probe_features(FakeModel(),object(),chunks,stats(),noise)
    # normalized zero=-1, x_t=-.9; mean positions 0..4=2.
    np.testing.assert_allclose(got,1.1)
    runtime=probe.probe_features(FakeModel(),object(),chunks.copy(),stats(),noise.copy())
    np.testing.assert_array_equal(got,runtime)
    assert not np.allclose(got,4.5-.9) # would be mean over all ten


def test_explicit_domain_noise_and_fail_closed_shapes():
    seed=probe.feature_probe_seed(7,3);a=probe.common_noise(seed);b=probe.common_noise(seed)
    np.testing.assert_array_equal(a,b);assert a.shape==(10,32)
    chunks=np.zeros((1,10,7),np.float32)
    for bad in (np.zeros((9,32)),np.full((10,32),np.nan)):
        with pytest.raises(ValueError): probe.probe_features(FakeModel(),object(),chunks,stats(),bad)
    with pytest.raises(ValueError): probe.probe_features(FakeModel(),object(),chunks,stats(),a,flow_time=.2)
    with pytest.raises(ValueError): probe.canonical_physical_chunks(np.zeros((1,9,7)),stats())


def test_entrypoint_preload_gates_before_model_initialization(monkeypatch,tmp_path):
    base=tmp_path/'base';norm=base/'assets/physical-intelligence/libero/norm_stats.json';norm.parent.mkdir(parents=True)
    norm.write_bytes(b'norm');(base/'params').mkdir()
    args=SimpleNamespace(base=base,base_sha256='base-ok',norm_sha256='norm-ok',expected_source_commit='source-ok')
    monkeypatch.setattr(cli.subprocess,'check_output',lambda *a,**k:'source-ok\n')
    monkeypatch.setattr(cli,'params_content_hash',lambda p:'base-ok')
    actual_norm=cli.hashlib.sha256(b'norm').hexdigest();args.norm_sha256=actual_norm
    monkeypatch.setattr(cli.jax,'default_backend',lambda :'gpu');monkeypatch.setattr(cli.jax,'devices',lambda kind=None:[object()])
    assert cli.validate_preload(args)[1]=='source-ok'
    called=[];monkeypatch.setattr(cli.bank,'initialize_native_base',lambda *a:called.append(True))
    args.base_sha256='wrong'
    with pytest.raises(ValueError,match='base'): cli.validate_preload(args)
    args.base_sha256='base-ok';args.norm_sha256='wrong'
    with pytest.raises(ValueError,match='norm'): cli.validate_preload(args)
    args.norm_sha256=actual_norm;monkeypatch.setattr(cli.jax,'default_backend',lambda :'cpu')
    with pytest.raises(RuntimeError,match='GPU'): cli.validate_preload(args)
    assert called==[]


def test_entrypoint_required_cli_fields():
    with pytest.raises(SystemExit): cli.parse_args([])
