:: This code defines a script which checks whether the code is running properly
:: on Hansel.

git -C %~dp0 pull
python "%~dp0\run_demonstrator.py"
