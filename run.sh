while [ 1 == 1 ]
do
    cp trained_models/a3c_super_mario_bros_1_1 test_models
    timeout 480 python test.py --saved_path test_models
    if read -r -N 1 -t 5; then
        break
    fi
done