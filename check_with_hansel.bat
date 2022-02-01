:: This code defines a script which checks whether the code is running properly
:: on Hansel.

conda activate demonstrator
git -C %~dp0 pull
python "%~dp0\run_demonstrator.py"
