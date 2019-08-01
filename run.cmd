call activate ptorch

cd C:\tensorflow1\Super-mario-bros-A3C-pytorch
:loop
copy trained_models\a3c_super_mario_bros_1_1 test_models /y
start python test.py --saved_path test_models
timeout /t 480
REM tasklist /V /FI "WindowTitle eq SuperMarioBrosRandomStagesEnv"
REM taskkill /F /FI "WindowTitle eq testmario*" /T   -- does not work
REM taskkill /F /FI "WindowTitle eq SuperMarioBrosRandomStagesEnv" /T
taskkill /F /FI "WindowTitle eq test.py" /T
goto loop