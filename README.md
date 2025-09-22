
# Transcription Translation App

App that does transcription and Translation using NPU with openvino runtime

## Installation

### Prerequisites
python>=3.10
Visual Studio Build Microsoft C++ Build Tools - https://visualstudio.microsoft.com/visual-cpp-build-tools/

### Clone repo
```bash
  git clone https://github.com/cmok1996/translation_npu.git
  cd translation_npu
```

### Install ovms
```bash
curl -L https://github.com/openvinotoolkit/model_server/releases/download/v2025.3/ovms_windows_python_on.zip -o ovms.zip
tar -xf ovms.zip
```

### Gather scripts
```bash
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/3/demos/common/export_models/export_model.py -o export_model.py
python -m venv ovms_venv
ovms_venv\Scripts\activate
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/3/demos/common/export_models/requirements.txt
pip install optimum-intel==1.24.0 optimum==1.26.1 --no-cache
mkdir models
```

### Export LLM
```bash
python export_model.py text_generation --source_model aisingapore/Llama-SEA-LION-v3-8B-IT --model_name aisingapore/Llama-SEA-LION-v3-8B-IT-4bit-group-sym-ov --target_device NPU --config_file_path models/config.json --model_repository_path models --overwrite_models --max_prompt_len 1024 --extra_quantization_params "--task text-generation-with-past --sym --ratio 1.0 --group-size 128" --ov_cache_dir ./models/.ov_cache
```

### Export whisper
```bash
optimum-cli export openvino --trust-remote-code --model openai/whisper-medium models/openai/whisper-medium
```

### Install project requirements
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Install ffmpeg
```bash
#run on powershell
winget install ffmpeg
```
## App launch

```bash
# run on powershell; initial run takes a longer time as it downloads whisper-small from whisper pipeline and kokoro model from huggingface
translation_demo.ps1
```


    

