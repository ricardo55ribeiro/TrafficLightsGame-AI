# TrafficLightsGame-AI

Q-learning agent that learns and plays **Alan Parr’s Traffic Lights**.

Trained Q-Tables are available here: **[Google Drive Folder](https://drive.google.com/drive/folders/13NguVbUBJ8DK68KiVXX9aep2TC-2lbAa?usp=sharing)**

---

## Game Rules

1. Components
   - A 3×4 rectangular grid board (3 rows, 4 columns).
   - Each cell can be in one of four states: Empty, Green, Yellow, Red.
2. Players
   - Two players alternate turns.
   - In simulations, Player 1 starts unless stated otherwise.
3. Starting Position
   - The game begins with all cells in the Empty state.
4. Turn Actions
   - On a turn, a player must select one cell and advance its state by exactly one step:
     - Empty → Green
     - Green → Yellow
     - Yellow → Red
   - Once a cell reaches Red, it cannot be changed further.
   - A turn cannot be skipped.
5. Winning Condition
   - A player immediately wins if their move results in a line of **three adjacent cells** sharing the **same non‑Empty color** (Green, Yellow, or Red).
   - A valid line may be:
     - Horizontal (across a row);
     - Vertical (down a column);
     - Diagonal (across adjacent rows and columns).
   - A line of four still qualifies if it includes at least one contiguous segment of three.

---

## Repository Map

- `TrafficLights_UI.py` — PvP Tkinter UI (local two-player match; highlights a winning line when it appears).
- `TrafficLights_QBot_Arena.py` — PvE Tkinter UI (play against the learned Q-bot). Looks for `qtable.json` in the working directory.
- `TrafficLights_TrainingCode.py` — Train, resume from checkpoint, or evaluate against baseline bots (Random / Alternate / Myopic). Periodically writes:
  - `qtable.json` — current policy
  - `*_checkpoint.json` — resumable state
  - `q*_deltas.jsonl` — incremental updates (for auditability)

License: MIT (see `LICENSE`).

---

## Results

(Work To Do)

---

## Notes

- Implementation uses **tabular Q-learning** with **ε-greedy** exploration.
- The board size is fixed by design at **3×4**.

---

## Requirements

- Python 3.x. No external pip dependencies.
- On Linux, install Tkinter if missing: `sudo apt install python3-tk`.
