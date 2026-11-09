# System One Model from Scratch

**Maintainer:** [Shifan T](https://github.com/shifanthottoli)

An independent reproduction of [CLM-8B](https://contrastive-lm.notion.site), an open System One model from Stanford and NVIDIA Research. This project includes training code and evaluation results.

A System One model does not write text. You give it a situation and a list of options, and it returns a probability for each option in one pass. [Jev](https://typesafe.ai) made the idea popular but is closed source, so this repo builds an open equivalent.

## The model

```text
Question + document -> Qwen3-8B (frozen) -> State head  (9.44M) -> 512-d
Answer              -> Qwen3-8B (frozen) -> Action head (9.44M) -> 512-d
```

We take the hidden state of the last token (4096 numbers), normalize it, and pass it through a small network: 4096 -> 1536 -> 1536 (LayerNorm) -> 512, with GELU. The score is the cosine of the two vectors times a learned scale. Only the two heads are trained, about 18.9M parameters.

## What was public, and what we rebuilt

The [official repo](https://github.com/Contrastive-LM/CLM) has the serving engine, a fine-tuning script and the T-Rex game. It does not have the pre-training or mid-training code. We recovered those recipes from the settings stored inside the authors' public checkpoints ([jackyk02/qwen3_8b_head](https://huggingface.co/jackyk02/qwen3_8b_head)) and wrote our own trainer.

| Stage | Data | Recipe | Script |
|---|---|---|---|
| Vectors | [62.45M Q&A vectors](https://huggingface.co/datasets/Contrastive-LM/CLM-v0.1-Pretrain-Nemotron), about 1 TB; [hard negatives](https://huggingface.co/datasets/jackyk02/nemotron_dqa_qwen3_8b_neg_embeddings), about 0.25 TB | Copy to a Modal volume | `pipeline/stage_data.py` |
| Pre-training | 621 chunks | Batch 16,384, lr 0.00653 constant, AdamW, no weight decay, temperature starts at 0.07 and is learned (scale capped at 100), same-document pairs masked | `pipeline/train_heads.py --stage pre` |
| Mid-training | 2.8M pairs with up to 10 hard negatives each | Batch 16,384, lr 0.001 constant, negatives added to the state-to-action direction, weight 1.0 | `pipeline/train_heads.py --stage mid` |
| Post-training | 847k agent decisions from [ADP v2](https://huggingface.co/datasets/neulab/adp-v2) | **Our reconstruction.** 60% agent steps, 40% replay with hard negatives, lr 0.0003, 400 steps, temperature frozen | `pipeline/build_agent_data.py`, `embed_agent.py`, `train_heads.py --stage post --replay-negs` |
| Verifier | 22,576 passing DeepSWE steps | Batch 512, seed 1234, the repo's own `finetune.py` | `pipeline/run_deepswe.py` |
| Testing | Held-out `shard_00/0028-0029` and `shard_05/0083-0084` | One code path for every head | `pipeline/final_eval.py`, `run_trex.py` |

## Run it

You need a [Modal](https://modal.com) account and about 1.3 TB of volume storage. No API keys are needed for the public data.

```bash
pip install modal
cd pipeline
modal run --detach stage_data.py
modal run --detach train_heads.py --stage pre --max-steps 30000
modal run --detach train_heads.py --stage mid --init /data/runs/pre/best.pt --run mid
modal run --detach build_agent_data.py
modal run --detach embed_agent.py
modal run --detach train_heads.py --stage post --init /data/runs/mid/best.pt --run post --max-steps 400 --lr 0.0003 --replay-negs
modal run run_deepswe.py --tag ours --init /data/runs/post/best.pt
modal run run_trex.py --tag ours --ckpt /data/runs/post/best.pt
```

Serving needs `vllm==0.11.0` with `transformers<5`. The unpinned install failed in our tests.

## Results

Measured on 6 November 2026, using 191,161 held-out pairs for the tricky question test.

| Test | Published | Authors' weights, our test | Our model |
|---|---|---|---|
| Tricky Q&A, after pre-training | 52.1% | 51.7% | **51.4%** |
| Tricky Q&A, after mid-training | 69.2% | 69.2% | **68.5%** |
| Tricky Q&A, after post-training | 68.5% | 68.6% | **69.0%** |
| DeepSWE, seed 1234 | 31/38 | 31/38 | **31/38** |
| DeepSWE, mean of 5 seeds | not published | 31.0 | **30.6** |
| T-Rex, median latency | 16.5 ms | 17.1 ms | **17.1 ms** |
| T-Rex, courses survived, shield on | 5/5 | 4/5 | **3/5** |
| T-Rex, planner agreement, shield on | 65.8% | 66.8% | **10%** |
| T-Rex, deaths over 5 courses, shield off | not published | 26 | **50** |

**The known gap:** the Q&A and DeepSWE results replicate, but our model reads the T-Rex labels far worse than the authors' head. The likely cause is post-training, whose full recipe and exact data are not public.
We also tried actions cut down to the bare decision, without the agent's narration (`pipeline/embed_actions.py`). In short 30-second game tests, planner agreement stayed at 16% to 29% with the shield on, against about 80% for the authors' head, so that did not close the gap.

We could not rerun WikiRacing, Super Mario, BFCL or Terminal-Bench, because their harnesses or candidate sets are not public. The Jev numbers we cite come from the authors' logs and charts. We did not run Jev ourselves.

## What we learned

1. **Read the checkpoints.** The settings stored in them reproduced the pre-training and mid-training curve.
2. **Replay needs hard negatives.** With plain Q&A replay, tricky-question accuracy fell to 59.8%. With hard negatives in the replay it held at 69.0%. The blog does not say what its replay contained, so this is our inference.
3. **The README fine-tune misses its own head.** The default dataset, with 405,919 steps, gave 28/38, the same as random. The 22,576 pairs in the head's metadata gave exactly 31/38, and matched the stored validation metric to every digit.
4. **The README's evaluation dataset returns 401.** We used the public copy, `tarsur385/deepswe-prm-embeddings-8k`.
5. **The seed matters.** On 38 tasks, the seed alone moved the score by up to 3 tasks.
6. **The safety shield does a lot of work.** The released head drops from 4/5 courses to 0/5 without it.

## Repo layout

```text
pipeline/   data copy, trainer, agent data, embedding, testing, game rendering
site/       presentation source and generated page
results/    measured numbers
```

## Credits and licences

CLM is by Jacky Kwok, Hangoo Kang, Tarun Suresh, Jon Saad-Falcon, Marco Pavone, Christopher Ré and Azalia Mirhoseini. Please cite [their work](https://contrastive-lm.notion.site). This independent reproduction was informed by a workshop presentation by Dr. Raj Dandekar and Vizuara AI Labs; it is not affiliated with them, the CLM authors, or TypeSafe. The Nemotron-derived data is under the NVIDIA Open Data License. The code in this repo is under the Apache 2.0 licence.
