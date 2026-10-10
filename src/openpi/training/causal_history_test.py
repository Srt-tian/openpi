from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from openpi.training import causal_history, plugin_data


def dataset():
    episodes=[]; tables={}
    for ep,task,length in ((0,0,4),(1,1,3)):
        e=plugin_data.Episode(ep,task,length,ep*100,ep*100+length,0,0,
          {k:(0,0,0.0,length/10) for k in plugin_data.VIDEO_KEYS})
        episodes.append(e)
        tables[ep]=plugin_data._EpisodeTable(
          state=np.arange(length*8,dtype=np.float32).reshape(length,8)+ep*1000,
          action=np.arange(length*7,dtype=np.float32).reshape(length,7)+ep*100,
          timestamp=np.arange(length)/10,frame_index=np.arange(length),
          index=np.arange(ep*100,ep*100+length),task_index=np.full(length,task))
    ds=plugin_data.LiberoV3Dataset('/unused',episodes,dict(enumerate(plugin_data.EXPECTED_TASKS)),10,10)
    ds._load_episode=mock.Mock(side_effect=lambda e:tables[e.index])
    ds._image=mock.Mock(return_value=np.zeros((256,256,3),np.uint8))
    return ds,tables


def stats(): return SimpleNamespace(q01=np.arange(7,dtype=np.float32),q99=np.arange(7,dtype=np.float32)+10)


def test_history_is_strictly_past_and_parent_target_unchanged():
    ds,tables=dataset(); wrapped=causal_history.CausalHistoryDataset(ds,stats(),2)
    parent=ds[3]; item=wrapped[3]; hist=item['causal_history']
    np.testing.assert_array_equal(item['actions'],parent['actions'])
    assert item['prompt']==parent['prompt']
    np.testing.assert_array_equal(hist['executed_action_raw'],tables[0].action[1:3])
    np.testing.assert_array_equal(hist['pre_state'],tables[0].state[1:3])
    np.testing.assert_array_equal(hist['post_state'],tables[0].state[2:4])
    assert item['causal_history_provenance']['truncated']
    assert item['causal_history_provenance']['omitted_past']==1
    assert item['causal_history_provenance']['burn_in_executed']==0
    assert not item['causal_history_provenance']['history_state_warmed']
    assert item['actions'].shape==(10,7) and np.all(item['actions']==tables[0].action[-1])


def test_episode_boundary_is_empty_and_normalization_is_exact_unclipped():
    ds,_=dataset(); item=causal_history.CausalHistoryDataset(ds,stats(),5)[4]
    assert item['causal_history']['mask'].shape==(0,)
    assert item['causal_history_provenance']['episode_index']==1
    value=np.array([[21,1,2,3,4,5,-6]],np.float32)
    expected=(value-stats().q01)/(stats().q99-stats().q01+1e-6)*2-1
    np.testing.assert_array_equal(causal_history.normalize_physical_actions(value,stats()),expected)
    assert causal_history.normalize_physical_actions(value,stats())[0,0]>1


def test_runtime_records_only_callbacks_for_executed_first_five():
    memory=causal_history.RuntimeExecutedHistory(5);memory.begin_episode('e')
    predicted=np.arange(70,dtype=np.float32).reshape(10,7)
    for step in range(5): memory.record_executed('e',step,np.zeros(8),predicted[step],np.ones(8))
    assert len(memory.last())==5
    np.testing.assert_array_equal(memory.last()[-1].action,predicted[4])
    with pytest.raises(ValueError,match='contiguous'): memory.record_executed('e',6,np.zeros(8),predicted[6],np.ones(8))
    with pytest.raises(ValueError,match='cross-episode'): memory.record_executed('other',5,np.zeros(8),predicted[5],np.ones(8))
    memory.begin_episode('next');assert memory.last()==()


def test_runtime_eviction_tracks_total_steps_and_zero_capacity():
    memory=causal_history.RuntimeExecutedHistory(2);memory.begin_episode('e')
    for step in range(8): memory.record_executed('e',step,np.zeros(8),np.full(7,step),np.ones(8))
    assert [row.step for row in memory.last()]==[6,7]
    snapshot=memory.last();snapshot[0].action[0]=999
    assert memory.last()[0].action[0]==6
    for bad in (7,9):
        with pytest.raises(ValueError,match='contiguous'):
            memory.record_executed('e',bad,np.zeros(8),np.zeros(7),np.ones(8))
    empty=causal_history.RuntimeExecutedHistory(0);empty.begin_episode('z')
    for step in range(4): empty.record_executed('z',step,np.zeros(8),np.zeros(7),np.ones(8))
    assert empty.last()==()
    with pytest.raises(ValueError): causal_history.RuntimeExecutedHistory(True)
