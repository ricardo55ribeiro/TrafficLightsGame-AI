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
   - A line of four still qualifies as it contains a segment of three (actually, two).

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

## How `TrafficLights_TrainingCode.py` Works

### 1) Learning Algorithm (Q‑Learning) and Exploration
This code implements **Tabular Q‑Learning** with an **ε‑greedy** policy. The agent maintains a value table $Q(s,a)$ estimating the long‑term utility of taking action $a$ in state $s$. During training, on each turn, it chooses:

- a **random legal move** with probability $\varepsilon$ (exploration), or
- the **greedy** action $\arg\max_a Q(s,a)$ with probability $1-\varepsilon$ (exploitation).

Exploration is necessary because, from an empty board, the action space is large and most states are unseen early on; without random tries the agent would get stuck repeating whatever it currently believes is the best. The script uses a **multiplicative decay** per game so that $\varepsilon$ starts high and decreases toward a floor $\varepsilon_{\text{end}}$: this yields broad exploration at first and stable exploitation later.  
**Evaluation** should be performed with $\varepsilon = 0$ (no randomness) so the agent plays its learned greedy policy deterministically; the training code’s evaluation options do exactly that.

**Update rule.** The current training loops apply **terminal‑only updates**: a Q‑value is adjusted **only when the game ends**. Let $r\in\{+1,0,-1\}$ denote the outcome from QBot’s perspective $(+1=\text{win}, -1=\text{loss}, 0=\text{draw/invalid})$. If QBot wins on move $(s,a)$, then

$$
Q(s,a) \leftarrow Q(s,a) + \alpha\, ( +1 - Q(s,a) ).
$$

If the **opponent** wins immediately after QBot’s last move $(\tilde{s},\tilde{a})$, that **pending** action is penalized:

$$
Q(\tilde{s},\tilde{a}) \leftarrow Q(\tilde{s},\tilde{a}) + \alpha\, ( -1 - Q(\tilde{s},\tilde{a}) ).
$$

No intermediate bootstrapping term is used (i.e., the target does not include \( \gamma \max_{a'} Q(s', a') \)); the update reduces to a Monte-Carlo-style move-credit on terminal feedback. To avoid first-move bias, the script alternates seats so QBot trains as both the first and second player across games.

### 2) Baseline Opponents and Training Curriculum
- **RandomBot** — samples uniformly among legal moves. It is ideal for **early training** because it exposes many different opening positions quickly. However, once QBot becomes competent, games against RandomBot rarely progress into rich mid/late positions, so it provides diminishing returns.
- **AlternateBot** — plays **myopically** (tries to win now, otherwise blocks an immediate loss) for an early, randomly chosen number of plays and then switches to **random** play. This curriculum **forces the game past the opening** so QBot experiences, explores, and updates on **late‑game** structures. The trade‑off is that, after switching to random, AlternateBot may **fail to punish** certain blunders, so QBot is not always corrected for every mistake.
- **MyopicBot** — a deterministic, tactical baseline: if a winning move exists **this turn**, it takes it; else if the opponent could win **next turn**, it blocks; otherwise it picks randomly among remaining legal moves. Training against MyopicBot provides **consistent punishment for obvious tactical errors** and requires QBot to spot immediate wins and blocks.

> Self‑play note: a QBot‑vs‑QBot mode is a natural extension for covering late‑game slips that baseline bots might miss and, in principle, it would double data throughput (both players learn). This training script focuses on the three bots above; self‑play can be added later as an optional mode.

### 3) Persistence: Atomic Saves, Delta Logging, and Checkpoints
- **Atomic Q‑table saves.** When writing `qtable.json`, the script first writes to a temporary file (e.g., `qtable.json.tmp`), flushes to disk, and then performs an **atomic rename**. If the process stops mid‑write, the previous `qtable.json` remains intact.
- **Delta logger.** Every Q‑value update is appended to a newline‑delimited JSON file (e.g., `q_deltas.jsonl`) as a compact record `{s, a, v}` containing the updated entry. On resume, the script **replays** all logged deltas into memory (or into an existing Q‑table) and then commits a single atomic save. After a successful consolidation, the delta file is **rotated** (e.g., renamed to `q_deltas.jsonl.done`) so it is not re‑applied.
- **Checkpoints and safe interrupt.** Periodically, and on Ctrl‑C, the script writes a checkpoint (e.g., `*_checkpoint.json`) storing the next game index and the current exploration rate $\varepsilon$. On interrupt, it also flushes any outstanding deltas and performs one last atomic save, so training can be **resumed exactly where it stopped** without losing Q‑table progress.

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
