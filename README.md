# Localized running for MiniGPT-4
A specialized implementation of Mini-GPT4 demo within command lines instead of Gradio online.

## Installation of [MiniGPT-4](https://minigpt-4.github.io/)
This section illustrates my personal experience in playing with the fascinating MiniGPT-4 (7b), big thanks to the authors.<br>
### Model Fetching with LFS
`curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash`<br>
`apt-get install git-lfs`<br>
`git lfs install`<br>
### Vicuna
`git clone https://huggingface.co/lmsys/vicuna-7b-delta-v1.1`
### Llama
`git clone https://huggingface.co/decapoda-research/llama-7b-hf`<br>
(Tips: A formal use should apply the checkpoints and tokenizer through [Llama](https://github.com/facebookresearch/llama/blob/main/README.md))
### Association V&L
Use the official FastChat tool to generate the final weights under directory FastChat<br>
`git clone https://github.com/lm-sys/FastChat.git`<br>
`pip3 install --upgrade pip`<br>
`pip3 install -e . `<i>dont't miss the '.'</i> <br>
`python -m fastchat.model.apply_delta --base {llama-13bOR7b-hf/}  --target {weights/}  --delta {vicuna-13bOR7b-delta-v1.1/}`

## Preparation for MiniGPT-4
### Env
`git clone https://github.com/Vision-CAIR/MiniGPT-4.git`<br>
`conda env create -f environment.yml`<br>
### Checkpoints
Download the checkpoints(e.g., [7b](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view))

## Final Step!
Claim your image file directory path and run:<br>
`conda activate minigpt4`<br>
`pip install -U bitsandbytes` to upgrade tp 0.38.1 <br>
`python demo_localized.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0 --img-dir {image directory}`<br>
It automatically reads all images in `img-dir` and input to Mini-GPT4 in turn. 
### Question Input
In `demo_localized.py`, the question is originally fixed for labeling on all images. Feel free to move `user_message` inside the for loop.

## Acknowledgments
Thanks to the whole community including authors of Llama, Vicuna, MiniGPT-4, and open-source contributors that helped me during the installation.

If you find this implementation helpful, please star and raise issues for further improvements together.


