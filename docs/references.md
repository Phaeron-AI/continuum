# References

**1. Mamba: Linear-Time Sequence Modeling with Selective State Spaces** — Gu & Dao, 2023 (arXiv:2312.00752). Your backbone. Read it for the selective SSM, the hardware-aware parallel scan (your `scan.py` is the pure-PyTorch version of their kernel), and the true Mamba block — input projection, depthwise causal conv, SiLU gate — which is exactly the `MambaMixer` you just added. This is the reference implementation your mixer is chasing.

**2. Transformers are Sample-Efficient World Models (IRIS)** — Micheli, Alonso & Fleuret, ICLR 2023 (arXiv:2209.00588). The closest architectural sibling to continuum: a discrete autoencoder tokenizes frames, then an autoregressive sequence model learns dynamics over those tokens. That *is* your Phase 1 → Phase 2 pipeline — continuum is essentially IRIS with Mamba swapped in for the Transformer. Read it for the overall recipe and how they use the world model for imagination/rollout.

**3. Diffusion Models Are Real-Time Game Engines (GameNGen)** — Valevski et al., Google, 2024 (arXiv:2408.14837). The most direct validation of what you're doing and where your #1 fix comes from. They hit exactly your problem — autoregressive drift over long play — and the fix is the one you implemented: corrupting context frames with noise during training so the model learns to correct its own errors. Read it for the drift mitigation and the real-time serving story (20 fps, stable over minutes). Note the contrast: they use diffusion, you use a token-based SSM — a legitimate axis of differentiation.

**4. Genie: Generative Interactive Environments** — Bruce et al., DeepMind, 2024 (arXiv:2402.15391). The playable-world-model vision your Phase 3 is aiming at. A spatiotemporal tokenizer + autoregressive dynamics model + a latent action model, trained from unlabelled video. Read it for action-conditioning design and the latent-action idea (relevant if you ever want to learn from action-free video), plus it's the clearest articulation of the "playable environment as a research artifact" framing.

**5. Finite Scalar Quantization: VQ-VAE Made Simple (FSQ)** — Mentzer et al., ICLR 2024 (arXiv:2309.15505). Your tokenizer. The 100% codebook usage you just saw (64/64) is the exact payoff this paper promises — FSQ sidesteps the codebook collapse that plagues VQ-VAE. Read it to understand why your Phase 1 was so clean and how to size `levels` when you scale the vocab up for richer frames.

Reading order I'd suggest: FSQ and Mamba first (your two components, and the shortest), then IRIS for the world-model recipe that ties them together, then GameNGen for the drift/real-time engineering, and Genie last for the ambition. If you want honorable mentions afterward — Ha & Schmidhuber's original *World Models* (2018) for the conceptual root, DIAMOND for the diffusion-world-model counterpoint, and the FiLM paper (Perez et al., 2018) for the conditioning layer you just added — say the word and I'll pull those too.

Sources:
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [Transformers are Sample-Efficient World Models (IRIS)](https://arxiv.org/abs/2209.00588)
- [Diffusion Models Are Real-Time Game Engines (GameNGen)](https://arxiv.org/abs/2408.14837)
- [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391)
- [Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/abs/2309.15505)