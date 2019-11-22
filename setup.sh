sudo apt update
sudo apt install python3-dev python3-pip
sudo pip3 install -U virtualenv
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt

