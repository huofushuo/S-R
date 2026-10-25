## Setup
```
conda create -n s-r python==3.9.0
conda activate s-r
pip install -r requirements.txt
```
comment off '@ torch.no_grad()' (llava/models/multimodal_encoder/clip_encoder/line40 to compute gradient) 
## Run
```bash
python main.py  #blip_models
```

```bash
python main_llava.py  #llava_models
```

## Responsible Use

This repository provides research code for reproducing the Self-Rambling (S-R) experiments and evaluating defenses against inference-cost attacks.

Use the code only on models and systems you own or have explicit permission to test. Do not use it to disrupt third-party services, exhaust shared resources, bypass service limits, or interfere with other users.

For serving experiments:
- Use an isolated test environment whenever possible.
- Set limits on the total number of requests, concurrency, and test duration.
- Monitor resource usage and stop the test if it affects users or systems outside the authorized scope.

Results depend on the model, hardware, and serving configuration. The findings reported in the paper do not imply that every deployment is equally vulnerable. Section VII discusses potential mitigations; their effectiveness should be assessed in the intended deployment.

If testing reveals a previously unreported vulnerability, contact the relevant provider or maintainer through its security reporting channel and coordinate disclosure before publishing deployment-specific details.
