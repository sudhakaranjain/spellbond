# SpellBond
Gym environment for the game Wordle. A simple action, state, and observation is included, with a random agent. Note that the spaces are very simplistic, and only included for a working example of the env.

### How to run
- Python >=3.8 required
- Create python virtual environment and install the `requirements.txt` file:
```commandline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
- From the parent folder, install SpellBond as a package:
```commandline
pip install -e .
```
To execute inference on full vocab, run the following command from `scripts` folder:
```commandline
cd scripts
python run.py --task infer
```
Optinally, to execute inference on partial vocab size, run the following:  
NOTE: Before executing, update the `words.txt` file to contain words only from this partial vocab list 
```commandline
python run.py --vocab-size {vocab_size} --task infer
```


