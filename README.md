## Setup
```
conda create -n s-r python==3.9.0
conda activate s-r
pip install -r requirements.txt
```
comment off '@ torch.no_grad()' (llava/models/multimodal_encoder/clip_encoder/line40 to compute gradient 
## Run
```bash
python main.py  #blip_models
```

```bash
python main_llava.py  #llava_models
```
