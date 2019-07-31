while [ 1 == 1 ]
do
    timeout 480 python test.py
    if read -r -N 1 -t 5; then
        break
    fi
done