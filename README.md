## Setup
```
conda create -n deco python==3.9
conda activate deco
pip install -r requirements.txt
```
commenting off '@ torch.no_grad()' (llava/models/multimodal_encoder/clip_encoder/line39
## Run
```bash
python main.py  #blip_models
```

```bash
python main_llava.py  #llava_models
```
