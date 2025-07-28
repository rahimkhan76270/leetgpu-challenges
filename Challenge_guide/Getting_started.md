# Creating New Challenges for LeetGPU

LeetGPU challenges are low-level GPU programming tasks focused on writing custom CUDA, Triton, or Tinygrad kernels. They evaluate both functional correctness and performance under real GPU constraints.

This guide provides detailed instructions for creating new GPU programming challenges for LeetGPU. It covers the complete process from concept to submission.

## Challenge Structure

Each challenge follows this directory structure:

```
challenges/<difficulty>/<number>_<name>/
├── challenge.html          # Problem description and examples
├── challenge.py           # Reference implementation and test cases
└── starter/              # Starter templates for each framework
    ├── starter.cu           # CUDA template
    ├── starter.mojo         # Mojo template
    ├── starter.pytorch.py   # PyTorch template
    ├── starter.tinygrad.py  # TinyGrad template
    └── starter.triton.py    # Triton template
```

## Creating the Challenge Files

### Step 1: Choose Your Challenge Location

1. Determine the appropriate difficulty level(easy, medium or hard)
2. Create your challenge directory: `challenges/<difficulty>/<name>/`

### Step 2: Create the Basic Structure

```bash
mkdir challenges/level_folder/your_challenge_name/
cd challenges/level_folder/your_challenge_name/
mkdir starter/
touch challenge.html challenge.py
touch starter/starter.cu starter/mojo starter/pytorch.py starter/tinygrad.py starter/triton.py
```

