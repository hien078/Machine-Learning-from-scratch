"""Unit tests for Phase 5 modern-AI modules (RL, generative, GNN, LLM, SSL)."""

import numpy as np
import pytest

from ml_first_principles.generative_models import (
    VAE,
    GANDiscriminator,
    GANGenerator,
    gan_discriminator_loss,
    gan_generator_loss,
    vae_elbo_loss,
)
from ml_first_principles.gnn_models import GATLayer, GCNLayer
from ml_first_principles.llm_models import BPETokenizer, LoRALinear, dpo_loss
from ml_first_principles.rl_models import (
    GOAL_REWARD,
    STEP_REWARD,
    TRAP_REWARD,
    GridWorldEnv,
    QLearningAgent,
)
from ml_first_principles.ssl_models import InfoNCELoss, PatchMasking

SENNRICH_CORPUS = " ".join(["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3)


def test_gridworld_rewards_and_termination():
    env = GridWorldEnv(grid_size=4, goal=(0, 1), trap=(2, 2))
    env.reset()
    state, reward, done, _ = env.step(3)  # Right onto the goal
    assert (state, reward, done) == (1, GOAL_REWARD, True)
    with pytest.raises(RuntimeError):
        env.step(0)

    env = GridWorldEnv(grid_size=4, goal=(3, 3), trap=(1, 0))
    env.reset()
    state, reward, done, _ = env.step(1)  # Down onto the trap
    assert (state, reward, done) == (4, TRAP_REWARD, True)

    env = GridWorldEnv(grid_size=4)
    env.reset()
    state, reward, done, _ = env.step(0)  # Up against the wall stays in place
    assert (state, reward, done) == (0, STEP_REWARD, False)


def test_gridworld_validates_layout():
    with pytest.raises(ValueError):
        GridWorldEnv(grid_size=4, goal=(4, 0))
    with pytest.raises(ValueError):
        GridWorldEnv(grid_size=4, goal=(1, 1), trap=(1, 1))
    with pytest.raises(ValueError):
        GridWorldEnv(grid_size=4, goal=(0, 0))


def test_qlearning_bellman_update():
    agent = QLearningAgent(
        num_states=2, num_actions=2, alpha=0.5, gamma=0.9, epsilon=0.0, random_state=0
    )
    agent.q_table[1] = [2.0, 0.0]
    td_error = agent.update(state=0, action=1, reward=1.0, next_state=1, done=False)
    assert np.isclose(td_error, 1.0 + 0.9 * 2.0)
    assert np.isclose(agent.q_table[0, 1], 0.5 * (1.0 + 0.9 * 2.0))
    # Terminal transitions drop the bootstrap term entirely.
    td_terminal = agent.update(state=0, action=0, reward=1.0, next_state=1, done=True)
    assert np.isclose(td_terminal, 1.0)


def test_qlearning_greedy_and_tie_breaking():
    agent = QLearningAgent(num_states=1, num_actions=3, epsilon=0.0, random_state=0)
    agent.q_table[0] = [0.0, 5.0, 0.0]
    assert agent.select_action(0) == 1
    agent.q_table[0] = [1.0, 1.0, 1.0]
    chosen = {agent.select_action(0) for _ in range(200)}
    assert chosen == {0, 1, 2}


def test_qlearning_learns_gridworld_policy():
    env = GridWorldEnv(grid_size=4)
    agent = QLearningAgent(num_states=16, num_actions=4, alpha=0.5, epsilon=0.2, random_state=0)
    for _ in range(300):
        state = env.reset()
        for _ in range(50):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
    # Greedy rollout from the start must reach the goal.
    agent.epsilon = 0.0
    state = env.reset()
    for _ in range(20):
        state, reward, done, _ = env.step(agent.select_action(state))
        if done:
            break
    assert done and reward == GOAL_REWARD


def test_vae_is_reproducible_with_valid_shapes():
    x = np.random.default_rng(0).random((4, 20))
    recon_a, mu_a, logvar_a = VAE(20, 8, 3, random_state=42).forward(x)
    recon_b, mu_b, logvar_b = VAE(20, 8, 3, random_state=42).forward(x)
    assert recon_a.shape == (4, 20) and mu_a.shape == (4, 3) and logvar_a.shape == (4, 3)
    assert np.allclose(recon_a, recon_b)
    assert np.allclose(mu_a, mu_b)
    assert np.allclose(logvar_a, logvar_b)
    assert np.all((recon_a > 0) & (recon_a < 1))


def test_vae_elbo_matches_closed_form():
    x = np.full((2, 4), 0.5)
    recon = np.full((2, 4), 0.5)
    mu = np.zeros((2, 3))
    logvar = np.zeros((2, 3))
    # At the prior (mu=0, logvar=0) the KL term is exactly zero and the BCE
    # of a uniform 0.5 reconstruction is 8 * log 2.
    assert np.isclose(vae_elbo_loss(recon, x, mu, logvar), 8.0 * np.log(2.0), rtol=1e-6)
    # Shifting every mean to one adds exactly 0.5 * sum(mu^2) = 3.0 of KL.
    mu_shifted = np.ones((2, 3))
    delta = vae_elbo_loss(recon, x, mu_shifted, logvar) - vae_elbo_loss(recon, x, mu, logvar)
    assert np.isclose(delta, 3.0, rtol=1e-6)


def test_gan_forward_and_losses():
    gen = GANGenerator(latent_dim=5, output_dim=6, random_state=0)
    disc = GANDiscriminator(input_dim=6, random_state=1)
    z = np.random.default_rng(2).standard_normal((8, 5))
    fake = gen.forward(z)
    assert fake.shape == (8, 6) and np.all((fake > -1) & (fake < 1))
    prob = disc.forward(fake)
    assert prob.shape == (8, 1) and np.all((prob > 0) & (prob < 1))
    # A perfect discriminator has near-zero loss; a confident generator too.
    assert gan_discriminator_loss(np.array([1.0]), np.array([0.0])) < 1e-6
    assert np.isclose(gan_generator_loss(np.array([np.exp(-2.0)])), 2.0)


def test_gcn_normalization_matches_hand_computation():
    layer = GCNLayer(in_features=3, out_features=3, random_state=0)
    layer.weight = np.eye(3)
    x = np.eye(3)
    adj = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    out = layer.forward(x, adj)
    deg = np.array([2.0, 3.0, 2.0])
    expected = (adj + np.eye(3)) / np.sqrt(np.outer(deg, deg))
    assert np.allclose(out, expected)


def test_gcn_does_not_double_count_existing_self_loops():
    x = np.random.default_rng(0).random((3, 3))
    adj = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    layer = GCNLayer(3, 4, random_state=1)
    assert np.allclose(layer.forward(x, adj), layer.forward(x, adj + np.eye(3)))


def test_gat_output_is_convex_combination_of_neighbors():
    # With identical node features every attention-weighted average must
    # reproduce the shared projected feature vector exactly.
    layer = GATLayer(in_features=4, out_features=2, random_state=0)
    x = np.tile(np.random.default_rng(1).random(4), (5, 1))
    upper = np.triu((np.random.default_rng(2).random((5, 5)) > 0.5).astype(float), 1)
    adj = upper + upper.T
    out = layer.forward(x, adj)
    assert np.allclose(out, np.tile(x[0] @ layer.weight, (5, 1)))


def test_gnn_rejects_bad_adjacency():
    layer = GCNLayer(3, 2, random_state=0)
    x = np.zeros((3, 3))
    with pytest.raises(ValueError):
        layer.forward(x, np.zeros((2, 2)))
    with pytest.raises(ValueError):
        layer.forward(x, -np.ones((3, 3)))


def test_bpe_learns_expected_first_merges():
    tokenizer = BPETokenizer(target_vocab_size=13)
    tokenizer.fit(SENNRICH_CORPUS)
    # ('e', 's') co-occurs in newest (6) + widest (3) = 9 times, the maximum.
    assert tokenizer.merges[0] == ("e", "s")
    assert tokenizer.merges[1] == ("es", "t")


def test_bpe_fit_is_idempotent():
    tokenizer = BPETokenizer(target_vocab_size=15)
    tokenizer.fit(SENNRICH_CORPUS)
    merges_first = list(tokenizer.merges)
    vocab_first = list(tokenizer.vocab)
    tokenizer.fit(SENNRICH_CORPUS)
    assert tokenizer.merges == merges_first
    assert tokenizer.vocab == vocab_first


def test_bpe_encode_decode_roundtrip():
    tokenizer = BPETokenizer(target_vocab_size=20)
    tokenizer.fit(SENNRICH_CORPUS)
    text = "lowest newest"
    tokens = tokenizer.encode(text)
    assert tokens and all(isinstance(token, str) for token in tokens)
    assert tokenizer.decode(tokens) == text
    with pytest.raises(RuntimeError):
        BPETokenizer().encode("low")


def test_lora_starts_at_base_and_validates_rank():
    lora = LoRALinear(in_features=6, out_features=4, r=2, random_state=0)
    x = np.random.default_rng(0).random((3, 6))
    # B is zero-initialized, so the adapted layer initially equals the base.
    assert np.allclose(lora.forward(x), x @ lora.W0)
    lora.lora_B += 1.0
    assert not np.allclose(lora.forward(x), x @ lora.W0)
    with pytest.raises(ValueError):
        LoRALinear(in_features=6, out_features=4, r=0)
    with pytest.raises(ValueError):
        LoRALinear(in_features=6, out_features=4, r=5)


def test_dpo_loss_reference_values():
    zero = np.zeros(3)
    # Zero margin: loss is exactly log 2 regardless of beta.
    assert np.isclose(dpo_loss(zero, zero, zero, zero, beta=0.5), np.log(2.0))
    # Huge negative margin: softplus(-beta*z) -> -beta*z without overflow.
    loss = dpo_loss(
        np.array([-50.0]), np.array([50.0]), np.array([0.0]), np.array([0.0]), beta=10.0
    )
    assert np.isfinite(loss) and np.isclose(loss, 1000.0)
    # Where the naive formula is stable, both must agree.
    rng = np.random.default_rng(3)
    pc, pr, rc, rr = rng.standard_normal((4, 5))
    naive = -np.mean(np.log(1.0 / (1.0 + np.exp(-0.1 * ((pc - pr) - (rc - rr))))))
    assert np.isclose(dpo_loss(pc, pr, rc, rr, beta=0.1), naive)


def test_infonce_matches_manual_reference():
    rng = np.random.default_rng(0)
    z1 = rng.standard_normal((4, 8))
    z2 = z1 + 0.1 * rng.standard_normal((4, 8))
    temperature = 0.5
    loss = InfoNCELoss(temperature=temperature).forward(z1, z2)

    def unit(v):
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    out = np.concatenate([unit(z1), unit(z2)], axis=0)
    sim = np.exp(out @ out.T / temperature)
    n = 4
    total = 0.0
    for i in range(2 * n):
        positive = sim[i, (i + n) % (2 * n)]
        denominator = np.sum(sim[i]) - sim[i, i]
        total += -np.log(positive / denominator)
    assert np.isclose(loss, total / (2 * n))


def test_infonce_is_finite_at_tiny_temperature():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((8, 16))
    loss = InfoNCELoss(temperature=0.001).forward(z, z + 0.01 * rng.standard_normal((8, 16)))
    assert np.isfinite(loss)
    with pytest.raises(ValueError):
        InfoNCELoss(temperature=0.0)


def test_patch_masking_restores_positions_and_dtype():
    masking = PatchMasking(mask_ratio=0.5, random_state=0)
    seq = np.random.default_rng(2).random((2, 10, 4)).astype(np.float32)
    kept, mask, restore = masking.mask_patches(seq)
    assert kept.shape == (2, 5, 4) and kept.dtype == np.float32
    assert mask.shape == (2, 10) and set(np.unique(mask)) <= {0.0, 1.0}
    assert np.all(mask.sum(axis=1) == 5)
    # Scattering the kept patches back must reproduce the visible positions.
    for b in range(2):
        padded = np.concatenate([kept[b], np.zeros((5, 4), dtype=np.float32)], axis=0)
        restored = padded[restore[b]]
        visible = mask[b] == 0
        assert np.allclose(restored[visible], seq[b, visible])


def test_patch_masking_validates_ratio():
    with pytest.raises(ValueError):
        PatchMasking(mask_ratio=1.0)
    with pytest.raises(ValueError):
        PatchMasking(mask_ratio=0.99).mask_patches(np.zeros((1, 10, 2)))
