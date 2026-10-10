from types import SimpleNamespace
import numpy as np
import pytest
from openpi.training import native_memory_features as features


def stats(width):return SimpleNamespace(q01=np.zeros(width),q99=np.ones(width))


def test_four_early_times_use_same_noise_and_official_velocity_sign():
    captured={}
    class Model:
        def flow_features(self,obs,x,time):
            captured.update(x=np.asarray(x),time=np.asarray(time))
            return np.zeros((4,10,32)),np.ones((4,10,1024))
    noise=features.cache_noise('train/episode:1/frame:5')
    actions=np.full((10,7),.25,np.float32)
    response=features.extract_training_features(Model(),{},actions,stats(7),noise)
    np.testing.assert_array_equal(captured['time'],np.asarray([1.,.9,.8,.7],np.float32))
    np.testing.assert_array_equal(captured['x'][0],noise)
    normalized=(actions/(1+1e-6))*2-1
    np.testing.assert_allclose(response['target_velocity7'][0],noise[:,:7]-normalized,rtol=1e-6,atol=1e-6)
    for i in range(1,4):np.testing.assert_array_equal(response['target_velocity7'][i],response['target_velocity7'][0])
    assert response['hidden'].shape==(4,10,1024)


def test_history_does_not_read_future_rows_and_keeps_current_poststate():
    table=SimpleNamespace(state=np.arange(48,dtype=np.float32).reshape(6,8),action=np.arange(42,dtype=np.float32).reshape(6,7))
    before,meta=features.episode_history(table,3,stats(7),stats(8))
    assert before.shape==(3,23) and meta['omitted_past']==0
    table.state[4:]=999;table.action[3:]=999
    after,_=features.episode_history(table,3,stats(7),stats(8))
    np.testing.assert_array_equal(before,after)
    empty,_=features.episode_history(table,0,stats(7),stats(8))
    assert empty.shape==(0,23)
    truncated,meta=features.episode_history(table,3,stats(7),stats(8),max_history=2)
    np.testing.assert_array_equal(truncated,before[1:])
    assert meta['omitted_past']==1 and meta['burn_in_executed']==0


def test_noise_is_domain_bound_and_preserves_global_rng():
    np.random.seed(123);before=np.random.get_state()
    first=features.cache_noise('x');second=features.cache_noise('x')
    np.testing.assert_array_equal(first,second)
    assert not np.array_equal(first,features.cache_noise('y'))
    after=np.random.get_state();assert before[0]==after[0]
    np.testing.assert_array_equal(before[1],after[1]);assert before[2:]==after[2:]
    assert features.schema('a'*64,'b'*64)['hidden_shape']==[4,10,1024]


def test_wrong_cfn_pooled_features_and_invalid_time_noise_are_rejected():
    class Bad:
        def flow_features(self,*args):return np.zeros((4,10,32)),np.zeros((4,1024))
    with pytest.raises(ValueError,match='response'):
        features.extract_training_features(Bad(),{},np.zeros((10,7)),stats(7),np.zeros((10,32)))
    with pytest.raises(ValueError,match='noise'):
        features.extract_training_features(Bad(),{},np.zeros((10,7)),stats(7),np.zeros((7,)))
