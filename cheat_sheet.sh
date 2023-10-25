source /p/scratch/ccstdl//dantuluri1/venv/bin/activate
ml --force purge
ml use $OTHERSTAGES
ml Stages/2023
ml GCC
ml OpenMPI
ml CUDA
ml cuDNN
ml NCCL
ml git
ml PyTorch
ml torchvision
module unload PyTorch 
module unload torchvision
export HF_HOME="/p/scratch/ccstdl/dantuluri1/transformers_cache"
srun --pty --nodes=1 -A cstdl --partition develbooster --gres gpu:4 /bin/bash