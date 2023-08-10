# Reinforcement Learning Tic-Tac-Toe

This program uses a [Q-Learning](https://en.wikipedia.org/wiki/Q-learning) algorithm to teach an AI to play [Tic-Tac-Toe](https://en.wikipedia.org/wiki/Tic-tac-toe).

## Program Requirements

* Python 3.11+

Python libraries:

* matplotlib

The required Python libraries can be installed using `pip3`:

```sh
pip3 install -r requirements.txt
```

## Program Usage

From the project directory, run the `main.py` file to run the simulation:

```sh
python3 main.py -p
```

The following command-line options are available:

```sh
usage: main.py [-h] [-g] [-p]

Train a Q-Learning Agent to play Tic-Tac-Toe.

options:
  -h, --help   show this help message and exit
  -g, --graph  generate Q-Learning Agent vs Random Agent graph
  -p, --play   play 10 rounds against the Q-Learning Agent
```

# References

* "Artificial Intelligence: A Modern Approach" by Peter Norvig and Stuart J. Russell
* "Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto
