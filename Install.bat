@echo off
cls
call python -m venv VirtualEnv
call VirtualEnv\Scripts\activate
call VirtualEnv\Scripts\python -m pip install -r requirements.txt
call VirtualEnv\Scripts\deactivate
exit
