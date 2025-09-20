# Start the second PowerShell window as administrator
Start-Process powershell -Verb RunAs -ArgumentList @"
    -NoExit -Command "cd 'C:\Users\aicoe\llm_openvino2025.2'; .\venv\Scripts\Activate.ps1; .\ovms\setupvars.ps1; ovms --model_path models/aisingapore/Llama-SEA-LION-v3-8B-IT-4bit-group-sym-ov --model_name aisingapore/Llama-SEA-LION-v3-8B-IT-4bit-group-sym-ov --port 9000 --rest_port 8000 --log_level DEBUG"
"@

Start-Process powershell -Verb RunAs -ArgumentList @"
    -NoExit -Command "cd 'C:\Users\aicoe\demos'; .\venv\Scripts\Activate.ps1; python translation/translate_app.py"
"@