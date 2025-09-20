# Start the second PowerShell window as administrator
Start-Process powershell -Verb RunAs -ArgumentList @"
    -NoExit -Command "cd '$PSScriptRoot'; .\ovms_venv\Scripts\Activate.ps1; .\ovms\setupvars.ps1; ovms --model_path models/aisingapore/Llama-SEA-LION-v3-8B-IT-4bit-group-sym-ov --model_name aisingapore/Llama-SEA-LION-v3-8B-IT-4bit-group-sym-ov --port 9000 --rest_port 8000 --log_level DEBUG"
"@

Start-Process powershell -Verb RunAs -ArgumentList @"
    -NoExit -Command "cd '$PSScriptRoot'; .\venv\Scripts\Activate.ps1; python translate_app.py"
"@