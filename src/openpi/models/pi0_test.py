import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.models.pi0 as _pi0
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


class _FakeLlm:
    def __call__(self, inputs, **kwargs):
        del kwargs
        prefix, suffix = inputs
        return (prefix + 0.25, suffix * 1.5 + 0.5), None


class _FakePi05:
    action_horizon = 2
    flow_features = _pi0.Pi0.flow_features
    compute_loss = _pi0.Pi0.compute_loss

    def __init__(self):
        self.PaliGemma = type("PaliGemma", (), {"llm": _FakeLlm()})()

    def embed_prefix(self, observation):
        del observation
        return jnp.ones((2, 1, 4)), jnp.ones((2, 1), dtype=bool), jnp.array([False])

    def embed_suffix(self, observation, x_t, time):
        del observation
        tokens = x_t[..., :4] + time[:, None, None]
        return tokens, jnp.ones((2, 2), dtype=bool), jnp.array([True, False]), jnp.zeros((2, 4))

    def action_out_proj(self, hidden):
        return jnp.concatenate((hidden, hidden), axis=-1)


def _old_loss_reference(model, rng, observation, actions, train):
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
    prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar, cond = model.embed_suffix(observation, x_t, time)
    input_mask = jnp.concatenate((prefix_mask, suffix_mask), axis=1)
    mask = _pi0.make_attn_mask(input_mask, jnp.concatenate((prefix_ar, suffix_ar), axis=0))
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, suffix), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=mask, positions=positions, adarms_cond=[None, cond]
    )
    velocity = model.action_out_proj(suffix[:, -model.action_horizon :])
    return jnp.mean(jnp.square(velocity - (noise - actions)), axis=-1)


def test_explicit_flow_features_and_compute_loss_parity(monkeypatch):
    model = _FakePi05()
    actions = jnp.arange(32, dtype=jnp.float32).reshape(2, 2, 8) / 10
    rng = jax.random.key(7)
    monkeypatch.setattr(_model, "preprocess_observation", lambda key, obs, train: obs)
    expected = _old_loss_reference(model, rng, object(), actions, True)
    actual = model.compute_loss(rng, object(), actions, train=True)
    np.testing.assert_array_equal(actual, expected)

    time = jnp.array([0.2, 0.8])
    velocity, hidden = model.flow_features(object(), jnp.ones_like(actions), time)
    assert hidden.shape == (2, 2, 4) and velocity.shape == actions.shape
    np.testing.assert_array_equal(velocity, model.action_out_proj(hidden))
