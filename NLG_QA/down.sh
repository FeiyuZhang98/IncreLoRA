

# IncreLoRA/NLU OR NLG
conda create -n incre_NLG python=3.7
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt

pip install -e . 
pip install -e ../loralib/
