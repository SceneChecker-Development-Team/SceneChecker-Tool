brew update
brew install gmp python-tk

python3 -m pip install --user --upgrade pip
python3 -m pip install --user --upgrade setuptools

# python3 -m pip install --user tk
python3 -m pip install --user numpy scipy matplotlib
python3 -m pip install --user dill
python3 -m pip install --user ray
python3 -m pip install --user cvxopt
python3 -m pip install --user polytope
python3 -m pip install --user PICOS
python3 -m pip install --user pypoman
python3 -m pip install --user torch torchvision torchaudio

# # Dependencies for Flow*
# brew install mpfr
# brew install gsl
# brew install glpk
# brew install bison 
# brew install flex

# cd flowstar_plot
# make clean
# make
# cd ..

# Dependencies for DryVR
python3 -m pip install --user networkx python-igraph sympy==1.6.2 z3-solver six


