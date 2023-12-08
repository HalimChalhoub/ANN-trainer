@echo off
call VirtualEnv\Scripts\activate
call python "%~dp0main.py"
call VirtualEnv\Scripts\deactivate
exit
