mamba create -n universal-unconsciousness -y python=3.11
source activate universal-unconsciousness
mamba install -y jupyter jupyterlab matplotlib numpy pandas scikit-learn scipy tqdm pip cython -c conda-forge
mamba install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install hydra-core wandb
pip install hydra-submitit-launcher --upgrade