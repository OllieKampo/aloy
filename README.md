# Aloy

The content of this repository is currently under development. Eventually, the aim is that this repository will contain a comprehensive suite of algorithms and data structures for artificial intelligence, machine learning, and autonomous robots.

Aloy will be split into 6 parts:
- Concurrency: Tools for concurrent multi-threaded programming and thread-safe synchronization.
- Control: Algorithms and systems for building control systems and controller, including GUIs for viewing control system responses.
- Data structures: Easy to use and efficient data structures for general programming problems and artificial intelligence.
- Games: Implementations of various simple 2D games with PySide6 based GUIs, all of which are compatible with Gymnasium for testing reinforcement learning algorithms.
- GUIs: Systems easing the development of complex GUI based applications with PySide6.
- Intelligence: Algorithms for artificial intelligence, with a focus on reinforcement learning and game playing.
- Optimization: Algorithms for optimization, with a focus on control systems optimisation.

## Current Ideas

- Re-implement thread-queues (including channeled queue) and heap-queue (in Rust).
- Add MultiClockThread.
- Lead-lag compensators for controllers,
- Control system demonstration and testing GUI,
    - Reimplement master's thesis design, but better!
- Genetic algorithms;
    - Support variable length chromosomes,
    - Support multiple-chromosomes.
- Particle swarm optimisation,
- The Bees algorithm,
- Optimisation test functions,
- Optimisation error surface plotting,
- Optimisation process viewer GUI,
- General reinforcement learning API,
    - Try Ray https://github.com/ray-project/ray and https://docs.ray.io/en/latest/rllib/index.html, and cleanRL https://github.com/vwxyzjn/cleanrl.
- General deep game tree and MCTS API,
- New event flow and command flow system with pub-sub style pattern for GUIs,
- New threadpool system and futures implementation,
- New thread-safe queues implementation,
- Snake game with GUI and RL,
    - Support saving options and highscores,
    - Support recoding games for use as replay and for immitation learning.
- Tetris game with GUI, curses, and RL,
    - Inspiration for GUI https://github.com/janbodnar/PyQt6-Tutorial-Examples, https://doc.qt.io/qt-6/qtwidgets-widgets-tetrix-example.html, and https://gitpress.io/u/1155/pyqt-example-tetrix,
    - Inspiration for curses https://github.com/cSquaerd/CursaTetra, https://github.com/orodley/curses-tetris/blob/master/main.py, https://codereview.stackexchange.com/questions/249326/python-3-curses-terminal-tetris, https://gitlab.com/mkhCurses/tetris-curses-python, https://github.com/adrienmalin/Terminis/tree/master.
- Pacman game with GUI and RL,
- Connect four game with GUI and deep game trees;
    - Take inspiration from http://blog.gamesolver.org/ and https://github.com/PascalPons/connect4.
    - Support playing over socket connection.
- Block breaker game using PyGame,
- Control interface for robots,
- Control and path planning simulator GUI for robots,
- More efficient data holders and multi-table data holders that you can attach to other classse and objects to capture performance statistics,
- Benchmarking tools.
