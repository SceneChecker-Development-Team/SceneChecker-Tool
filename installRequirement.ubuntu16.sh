# Install Python3.7
sudo apt update
sudo apt install software-properties-common -y
sudo apt install git -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.7 python3.7-dev python3.7-dbg python3.7-tk python3.7-tk-dbg -y
sudo apt install python3-pip -y

# Install dependencies for SceneChecker
sudo apt update
sudo apt install libgmp3-dev -y

sudo python3.7 -m pip install --upgrade pip
sudo python3.7 -m pip install --upgrade setuptools

python3.7 -m pip install --user numpy scipy matplotlib
python3.7 -m pip install --user dill
python3.7 -m pip install --user ray
python3.7 -m pip install --user cvxopt
python3.7 -m pip install --user polytope
python3.7 -m pip install --user PICOS
python3.7 -m pip install --user pypoman
python3.7 -m pip install --user torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# Dependencies for Flow*
sudo apt install libmpfr-dev -y
sudo apt install libgsl-dev -y
sudo apt install libglpk-dev -y
sudo apt install bison -y
sudo apt install flex -y

cd flowstar_plot
make clean
make
cd ..

# Dependencies for DryVR
sudo apt install libcairo2-dev libgraphviz-dev python3-cairo python3-tk python3-pygraphviz -y
python3.7 -m pip install --user pycairo==1.19.1 \
pygraphviz \
glpk \
networkx \
python-igraph \
matplotlib \
sympy \
z3-solver \
six 


