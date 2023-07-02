# Localized running for Mini-GPT4
A specialized implementation of Mini-GPT4 demo within command lines instead of Gradio online.

## Installation of [Mini-GPT4](https://minigpt-4.github.io/)
This section illustrates my personal experience in playing with the fascinating Mini-GPT4 (7b), big thanks to the authors.<br>
### Model Fetching with LFS
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
apt-get install git-lfs
git lfs install
### Vicuna
`git clone https://huggingface.co/lmsys/vicuna-7b-delta-v0`
### Llama
`git clone https://huggingface.co/decapoda-research/llama-7b-hf`<br>
(Tips: A formal use should apply the checkpoints and tokenizer through [Llama](https://github.com/facebookresearch/llama/blob/main/README.md))
### Association V&L
Use the official FastChat tool to generate the final weights under directory FastChat<br>
`git clone https://github.com/lm-sys/FastChat.git
cd FastChat/
pip3 install --upgrade pip
pip3 install -e . #<i>dont't miss the '.'</i>
python -m fastchat.model.apply_delta --base /path/to/llama-13bOR7b-hf/  --target /path/to/save/working/vicuna/weight/  --delta /path/to/vicuna-13bOR7b-delta-v0/
`
## Preparation for Mini-GPT4
### Env
`git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4`
### Checkpoints
Download the checkpoints(e.g., [7b](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view))

## Claim your image file directory path and run:<br>
`python demo_localized.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0 --img-dir {image directory}`


