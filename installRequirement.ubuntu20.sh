# sudo apt update
# sudo apt install software-properties-common -y
# sudo apt install git -y
# sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install -y python3-pip python3-pil python3-pil.imagetk
sudo apt-get install -y libgmp3-dev

sudo python3 -m pip install --upgrade pip
sudo python3 -m pip install --upgrade setuptools

python3 -m pip install --user numpy scipy matplotlib
python3 -m pip install --user dill
python3 -m pip install --user ray
python3 -m pip install --user cvxopt
python3 -m pip install --user polytope
python3 -m pip install --user PICOS
python3 -m pip install --user pypoman
python3 -m pip install --user torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# Dependencies for Flow*
sudo apt-get install -y libmpfr-dev
sudo apt-get install -y libgsl-dev
sudo apt-get install -y libglpk-dev
sudo apt-get install -y bison
sudo apt-get install -y flex

cd flowstar_plot
make clean
make
cd ..

# Dependencies for DryVR
sudo apt-get install -y libcairo2-dev libgraphviz-dev python3-cairo python3-tk python3-pygraphviz
python3 -m pip install --user pycairo==1.19.1 \
pygraphviz \
glpk \
networkx \
python-igraph \
matplotlib \
sympy \
z3-solver \
six 


