## Streaming Simulation and Evaluation

### Streaming Simulation Scripts

- Run `simulate_streaming_qwen2_eval.py` to simulate **streaming inference with short utterances**.
- Run `simulate_streaming_qwen2_eval_longform.py` to simulate **streaming inference with long-form utterances**.

---

### SimulEval-Based Streaming Evaluation

For full streaming evaluation with latency metrics, use `s2s_st_agent.py` together with **SimulEval**.

This setup relies on a custom fork of SimulEval (my `maal` branch): https://github.com/AmirHussein96/SimulEval/tree/maal

#### Example SimulEval command

```bash
simuleval --agent s2s_st_agent.py \
          --source-segment-size 80 \
          --latency-metrics LAAL MAAL \
          --ctm-path /path/to/ctm \
          --t2t-align-path /path/to/awesome-align.{ids|out|parallel} \
          --source source.txt \
          --target target.txt \
          --output output \
          --quality-metrics BLEU \
          --config-path /path/to/exp_config.yaml \
          --model-path /path/to/checkpoints/stepXXXX.ckpt
```

## Reference SimulEval Example

A complete working example is available here:  
https://github.com/AmirHussein96/SimulEval/tree/maal/examples/speech_to_text

---

## Awesome-Align Input Files

When using **MAAL** with SimulEval, the following Awesome-Align files are required:

### `awesome-align.ids`
```text
maal_test.wav
```

### `awesome-align.out`
```text
8-8 2-4 13-14 7-7 4-1 9-9 3-0 11-12 10-10 13-13 0-5 6-6 1-3 12-14 5-2
```
### `awesome-align.parallel`
```text
Fue su reino, en su tiempo, el más poderoso e influyente de Europa Occidental. |||
At that time, his kingdom was the most powerful and influential kingdom of Occidental Europe.
```
These alignment files are used to compute MAAL and other alignment-aware latency metrics during streaming evaluation.

## Quality Evaluation

To run the quality evaluation, execute the following command:

```bash
python s2s/scripts/score_st.py --json <path-to-generated-results.json>
```