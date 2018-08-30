echo "Torch 7 & LuaJIT Auto-Installation Script"
echo "------- Written by Luke Rose 2018 -------"
echo ""

echo "Moving to user directory..."
cd ~

echo "Removing old Torch files..."
yes | sudo rm -r torch

echo "Pulling latest version of Torch..."
git clone https://github.com/torch/distro.git ~/torch --recursive

echo "Cleaning any old installations of Torch..."
cd ~/torch
./clean.sh

echo "Installing required packages..."
install-deps

echo "Installing Torch..."
export TORCH_NVCC_FLAGS="-D__CUDA_NO_HALF_OPERATORS__"
./install.sh

echo "Adding Torch to environment path"
export PATH="${PATH}:~/torch/install/bin"

echo "Activating Torch"
torch-activate

echo "Installing Torch CLI..."
luarocks install image

echo "Installing Torch debug tools..."
luarocks install mobdebug
