Here are the steps to run the project:
```shell
git clone --depth 1 https://github.com/EnsueCollectR/LLaMA-Factory
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
llamafactory-cli webui
```
This would bring up the llama-factory web interface for training/testing   
Framework's readme in https://github.com/EnsueCollectR/LLaMA-Factory/blob/main/Framework_README.md
