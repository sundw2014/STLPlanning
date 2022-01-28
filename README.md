# Multi-agent Motion Planning from Signal Temporal Logic Specifications
The full paper is available at https://arxiv.org/abs/2201.05247

## Requirements
Please install the Gurobi optimizer by following the instructions on the official website https://www.gurobi.com/products/gurobi-optimizer/
You might be eligible for a free academic license https://www.gurobi.com/academia/academic-program-and-licenses/

Then install the following Python packages.
```
pip install numpy matplotlib pypoman
```

If you get an error about the ```gmp.h``` file when installing pypoman, install the gmp libary. If you are using Ubuntu, run
```
sudo apt-get install libgmp-dev
```
Then, rerun the above ```pip``` command.

## Usage
Just run each python script to plan a path and visualize it. For example,
```
python run_stlcg-1.py
```
