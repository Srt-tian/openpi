import numpy as np
import pytest
import serve_pi05_memory_features as service


def test_training_feature_service_preserves_payload_and_has_no_policy_actions(monkeypatch):
    policy=service.MemoryFeaturePolicy(object(),lambda x:{'state':x['observation/state']},object())
    monkeypatch.setattr(service.model_api.Observation,'from_dict',lambda x:x)
    monkeypatch.setattr(service.model_api,'preprocess_observation',lambda *a,**k:a[1])
    def features(model,obs,actions,stats,noise):
        assert obs['state'].shape==(4,8)
        assert actions.shape==(10,7) and noise.shape==(10,32)
        return {'hidden':np.ones((4,10,1024)), 'base_velocity':np.zeros((4,10,32)),
                'target_velocity7':np.zeros((4,10,7)), 'time':np.array([1.,.9,.8,.7])}
    monkeypatch.setattr(service.feature,'extract_training_features',features)
    payload={'observation/image':np.zeros((2,2,3),np.uint8),
        'observation/wrist_image':np.zeros((2,2,3),np.uint8),'observation/state':np.zeros(8),
        'prompt':'complete task instruction','actions7':np.zeros((10,7)),'noise32':np.zeros((10,32))}
    keys=set(payload);output=policy.infer(payload)
    assert set(payload)==keys
    assert set(output)=={'hidden','base_velocity','target_velocity7','time'}
    assert 'actions' not in output
    with pytest.raises(ValueError,match='schema'):
        policy.infer({k:v for k,v in payload.items() if k!='actions7'})
