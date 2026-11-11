@echo off
REM VisioVetor — setup + execução local (Windows)
REM Defina SKIP_FINETUNE=1 para pular o fine-tuning.
setlocal
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
  echo Criando ambiente virtual...
  python -m venv venv || goto :erro
)
set PY=venv\Scripts\python.exe

%PY% -c "import torch, transformers, gradio, datasets" 2>nul
if errorlevel 1 (
  echo Instalando dependencias...
  %PY% -m pip install --upgrade pip
  where nvidia-smi >nul 2>nul || %PY% -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  %PY% -m pip install -r requirements.txt || goto :erro
)

if not exist data\raw\captions.jsonl (
  echo Baixando dataset...
  %PY% scripts\prepare_data.py || goto :erro
)

if not exist models\ptbr-finetuned\config.json if not "%SKIP_FINETUNE%"=="1" (
  echo Fine-tuning...
  %PY% scripts\finetune.py || goto :erro
)

%PY% scripts\build_index.py || goto :erro
echo Abrindo interface em http://127.0.0.1:7860
%PY% app.py
goto :eof

:erro
echo Falhou. Veja a mensagem acima.
exit /b 1
