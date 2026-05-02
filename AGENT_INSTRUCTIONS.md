# AI Agent Instructions

## Python Environment
When executing Python code, running scripts, or installing dependencies in this workspace, **you MUST use the `myenv` miniconda environment.**

### How to use the environment
If you are running a single command via the terminal, use `conda run`:
```bash
conda run -n myenv python script.py
```

If you are using a persistent shell or need to run multiple commands, activate the environment first:
```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate myenv
```
