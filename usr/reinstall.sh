sudo apt-key add Repo.keys
sudo cp -R sources.list* /etc/apt/
sudo apt-get update
sudo apt-get install dselect
sudo dselect update
sudo dpkg --set-selections < Package.list
sudo apt-get dselect-upgrade -y
sudo apt clean && sudo apt autoclean
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
apt install -y openssh-server
rm -f /etc/ssh/sshd_config && cp sshd_config /etc/ssh/sshd_config
service ssh start
