# Imports
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import random
import json
import os
import tkinter as tk


# Defining Cell States as Integers (Easier Incrementation, according to Game's Logic)
EMPTY, GREEN, YELLOW, RED = 0, 1, 2, 3

@dataclass
class TrafficLightsGame:
    """Traffic Lights Game Logic"""
    rows: int = 3
    cols: int = 4
    board: List[List[int]] = field(default_factory=lambda: [[EMPTY]*4 for _ in range(3)])
    finished: bool = False
    winning_color: Optional[int] = None                    # Winning Colour
    winning_line: Optional[List[Tuple[int, int]]] = None   # Coordinates of the Winning Line

    current_player: int = 1                                # Two-Players Logic; Player 1 Starts
    winner: Optional[int] = None                           # Crown the Winner


    # Public API
    def advance(self, r: int, c: int) -> bool:
        """Advance the Cell at (r, c) by Exactly One State.
        Returns True if the Move was Successful; False if it was Invalid.
        Allowed Values: r in [0, 1, 2], c in [0, 1, 2, 3].
        """
        if self.finished:
            return False

        # Check Bounds
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False

        # Doesn't Allow RED to be Incremented
        if self.board[r][c] == RED:
            return False

        # Valid Move -> Advance Exactly One Step
        self.board[r][c] += 1

        # Switch Player after a Successful Move (P1->P2; P2->P1)
        self.current_player = 3 - self.current_player

        # Check for a Win After the Move
        won, color, line = self._check_win()
        if won:
            self.finished = True
            self.winning_color = color
            self.winning_line = line
            # Since we Already Switched, the Winner is the Previous Player
            self.winner = 3 - self.current_player
        
        return True


    # Verify if the Board has any Errors by Validating Coords
    def at(self, r: int, c: int) -> int:
        """Returns the Numeric State at (r, c)."""
        self._validate_coords(r, c)
        return self.board[r][c]


    def reset(self) -> None:
        """Resets the Game to the Initial (Empty) State."""
        self.board = [[EMPTY for _ in range(self.cols)] for _ in range(self.rows)]
        self.finished = False
        self.winning_color = None
        self.winning_line = None
        self.current_player = 1
        self.winner = None


    # Helper Functions
    def _validate_coords(self, r: int, c: int) -> None:
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise ValueError(f"Invalid Coords: ({r}, {c}) Out of Bounds!")
        

    def _check_win(self) -> Tuple[bool, Optional[int], Optional[List[Tuple[int, int]]]]:
        """Return (won, color, line_coords); color/line_coords are None if not won."""
        # Check All Length-3 Segments that Exist on a 3x4 Board.

        # Horizontal Segments
        for r in range(self.rows):
            for c in (0, 1):
                coords = [(r, c), (r, c + 1), (r, c + 2)]
                color = self._same_non_empty(coords)
                if color is not None:
                    return True, color, coords

        # Vertical Segments
        for c in range(self.cols):
            coords = [(0, c), (1, c), (2, c)]
            color = self._same_non_empty(coords)
            if color is not None:
                return True, color, coords

        # Diagonals (Down-Right Direction: \)
        for c in (0, 1):
            coords = [(0, c), (1, c + 1), (2, c + 2)]
            color = self._same_non_empty(coords)
            if color is not None:
                return True, color, coords

        # Diagonals (Down-Left Direction: /)
        for c in (2, 3):
            coords = [(0, c), (1, c - 1), (2, c - 2)]
            color = self._same_non_empty(coords)
            if color is not None:
                return True, color, coords

        return False, None, None


    def _same_non_empty(self, coords: List[Tuple[int, int]]) -> Optional[int]:
        """If all Coords Share the Same Non-Empty State, Return that State; else None."""
        r0, c0 = coords[0]
        color = self.board[r0][c0]
        if color == EMPTY:
            return None
        for (r, c) in coords[1:]:
            if self.board[r][c] != color:
                return None
        return color



# Q-Learning Agent
class QBot:
    """
    Tabular Q-learning Agent for Traffic Lights Game.
    - State Key Encodes Board and the Side-to-Move;
    - Action is (r, c);
    - No Learning During Arena Play, Simply Reads Q-Values from QTable
    """
    def __init__(
        self,
        seed: Optional[int] = None,
        q_path: Optional[str] = None
    ):
        self._rng = random.Random(seed)
        self.Q: dict[str, dict[str, float]] = {}
        self.q_path = q_path
        if q_path and os.path.exists(q_path):
            self.load(q_path)


    def candidate_moves(self, game: TrafficLightsGame) -> List[Tuple[int, int]]:
        if game.finished:
            return []
        return [
            (r, c)
            for r in range(game.rows)
            for c in range(game.cols)
            if game.board[r][c] != RED
        ]


    def choose(self, game: TrafficLightsGame) -> Optional[Tuple[int, int]]:
        """Best Choice from Q-table; Returns (r, c) or None if no Moves."""
        moves = self.candidate_moves(game)
        if not moves:
            return None

        s = self._state_key(game)
        # ensure state bucket
        if s not in self.Q:
            self.Q[s] = {}

        # Exploit: Pick Max-Q Move
        best_val = None
        best_moves = []
        for (r, c) in moves:
            a = self._action_key(r, c)
            val = self.Q[s].get(a, 0.0)
            if (best_val is None) or (val > best_val):
                best_val = val
                best_moves = [(r, c)]
            elif val == best_val:
                best_moves.append((r, c))

        # Survival-Bias (Prefer Moves that don’t Allow an Immediate Opponent Win)
        if len(best_moves) > 1:
            safe_moves = []
            for (r, c) in best_moves:
                # Simulate this Move
                tmp = TrafficLightsGame(rows=game.rows, cols=game.cols)
                tmp.board = [row[:] for row in game.board]
                tmp.current_player = game.current_player
                tmp.advance(r, c)

                # Check if Opponent has an Immediate Win
                opponent_moves = [
                    (rr, cc)
                    for rr in range(tmp.rows)
                    for cc in range(tmp.cols)
                    if tmp.board[rr][cc] != RED
                ]
                immediate_loss = False
                for (rr, cc) in opponent_moves:
                    sim = TrafficLightsGame(rows=tmp.rows, cols=tmp.cols)
                    sim.board = [row[:] for row in tmp.board]
                    sim.current_player = tmp.current_player
                    sim.advance(rr, cc)
                    if sim.finished and sim.winner == tmp.current_player:
                        immediate_loss = True
                        break

                if not immediate_loss:
                    safe_moves.append((r, c))

            if safe_moves:
                return self._rng.choice(safe_moves)

        # Uniformly Random Choice Among Equal Rated Moves
        return self._rng.choice(best_moves)


    def load(self, path: Optional[str] = None):
        path = path or self.q_path or "qtable.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.Q = {str(s): {str(a): float(v) for a, v in adict.items()} for s, adict in data.items()}


    # Key Helpers
    @staticmethod
    def _state_key(game: TrafficLightsGame) -> str:
        rows = [''.join(str(x) for x in row) for row in game.board]
        return '/'.join(rows) + f'|{game.current_player}'

    @staticmethod
    def _action_key(r: int, c: int) -> str:
        return f"{r},{c}"
    
    





# Play vs QBot: Tkinter UI
STATE_COLORS = {
    EMPTY:  "#ffffff",
    GREEN:  "#2ecc71",
    YELLOW: "#f1c40f",
    RED:    "#e74c3c",
}

STATE_TO_NAME = {EMPTY: "Empty", GREEN: "Green", YELLOW: "Yellow", RED: "Red"}

# Day/Night Theme Palettes
THEMES = {
    "day": {
        "canvas_bg": "#f7f7f7",
        "outline": "#333333",
        "text_fg": "#000000",
        "window_bg": "#f0f0f0",
        "bar_bg": "#f0f0f0",
    },
    "night": {
        "canvas_bg": "#000000",
        "outline": "#ffffff",
        "text_fg": "#ffffff",
        "window_bg": "#1a1a1a",
        "bar_bg": "#1a1a1a",
    },
}


class QBotUI(tk.Tk):
    """
    GUI to Play a Game Against Q-Learning QBot:
    - Choose P1 or P2, then Click New Game to Start;
    - Uses qtable.json to Load Learned Values and Decide the Best Move.
    """
    def __init__(self, qtable_path: str = "qtable.json", cell_size: int = 200):
        super().__init__()
        self.title("Traffic Lights Game, by Alan Parr")
        self.resizable(True, True)

        # Engine
        self.game = TrafficLightsGame()
        self.qtable_path = qtable_path
        self.qbot = QBot(q_path=qtable_path if os.path.exists(qtable_path) else None)

        self.cell = cell_size
        self.rect_ids = [[None]*self.game.cols for _ in range(self.game.rows)]

        self.theme = "day"                          # Theme is day by default
        self.board_frame_ids = None

        # Controls Bars
        self.bar = tk.Frame(self, padx=10, pady=8, bg=THEMES[self.theme]["bar_bg"])
        self.bar.pack(fill="x")

        # Player Choice
        self.you_label = tk.Label(self.bar, text="You:", font=("TkDefaultFont", 10, "bold"))
        self.you_label.pack(side="left", padx=(0,6))
        self.human_player_var = tk.IntVar(value=1)  # You are P1 by default
        
        # Player Circle Colour
        self.rb1 = tk.Radiobutton(self.bar, text="P1", variable=self.human_player_var, value=1, selectcolor="#1e90ff")
        self.rb1.pack(side="left")
        self.rb2 = tk.Radiobutton(self.bar, text="P2", variable=self.human_player_var, value=2, selectcolor="#1e90ff")
        self.rb2.pack(side="left", padx=(4,10))

        # Buttons
        self.newgame_btn = tk.Button(self.bar, text="New Game", command=self.new_game)
        self.newgame_btn.pack(side="left", padx=(6,6))

        # Theme Toggle Button
        self.theme_btn = tk.Button(self.bar, text="🌞", command=self.toggle_theme, bd=0, highlightthickness=0)
        self.theme_btn.pack(side="right")

        # Status and Footer
        self.status_var = tk.StringVar()
        self.status = tk.Label(self, textvariable=self.status_var)
        self.status.pack(padx=10, pady=(0,8), anchor="w")

        self.footer = None

        # Board
        w = self.game.cols * self.cell
        h = self.game.rows * self.cell
        self.canvas = tk.Canvas(self, width=w, height=h, bg=THEMES[self.theme]["canvas_bg"], highlightthickness=0)
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_click)

        # Layout
        EXTRA_W = 240
        TOPBAR_PAD = 150
        self.geometry(f"{w + EXTRA_W}x{h + TOPBAR_PAD}")
        self.state('zoomed') 

        self._init_cells()
        self._draw_board_frame()
        self._redraw_all()
        self._set_status("Pick P1 or P2, then Press New Game.")

        self.apply_theme()


    # Drawing
    def _init_cells(self):
        for r in range(self.game.rows):
            for c in range(self.game.cols):
                x0 = c * self.cell
                y0 = r * self.cell
                x1 = x0 + self.cell
                y1 = y0 + self.cell
                rect = self.canvas.create_rectangle(
                    x0, y0, x1, y1,
                    fill=self._cell_fill(EMPTY),
                    outline=self._outline_color(),
                    width=1
                )
                self.rect_ids[r][c] = rect

    def _draw_board_frame(self):
        """
        Draw an Extra frame on the Right and Bottom Edges so the Outer Border is Visible
        """
        w = self.game.cols * self.cell
        h = self.game.rows * self.cell
        color = self._outline_color()
        
        if self.board_frame_ids:
            for _id in self.board_frame_ids:
                self.canvas.delete(_id)
        
        right = self.canvas.create_line(w-1, 0, w-1, h-1, fill=color, width=1)
        bottom = self.canvas.create_line(0, h-1, w-1, h-1, fill=color, width=1)
        self.board_frame_ids = (right, bottom)

    def _update_cell(self, r: int, c: int):
        state = self.game.board[r][c]
        self.canvas.itemconfigure(self.rect_ids[r][c],
                                  fill=self._cell_fill(state),
                                  outline=self._outline_color(),
                                  width=1)


    def _redraw_all(self):
        for r in range(self.game.rows):
            for c in range(self.game.cols):
                self._update_cell(r, c)
        self._draw_board_frame()
        if self.game.finished:
            self._highlight_win()


    def _highlight_win(self):
        if not self.game.finished or not self.game.winning_line:
            return
        for (r, c) in self.game.winning_line:
            self.canvas.itemconfigure(self.rect_ids[r][c], outline="#1e90ff", width=4)


    def _set_status(self, msg: str):
        self.status_var.set(msg)

    # Theme Helpers
    def _outline_color(self):
        return THEMES[self.theme]["outline"]

    def _cell_fill(self, state: int):
        if self.theme == "night" and state == EMPTY:
            return "#000000"
        return STATE_COLORS[state]

    def toggle_theme(self):
        # Flip Theme and Icon
        self.theme = "night" if self.theme == "day" else "day"
        self.theme_btn.config(text="🌙" if self.theme == "night" else "🌞")
        self.apply_theme()

    def apply_theme(self):
        t = THEMES[self.theme]

        # Window + Top Bar
        self.configure(bg=t["window_bg"])
        self.bar.configure(bg=t["bar_bg"])

        # Top Controls
        for w in (self.you_label, self.rb1, self.rb2, self.newgame_btn, self.theme_btn):
            try:
                w.configure(bg=t["bar_bg"], fg=t["text_fg"], activeforeground=t["text_fg"])
            except tk.TclError:
                try:
                    w.configure(bg=t["bar_bg"], fg=t["text_fg"])
                except tk.TclError:
                    pass

        # Status/Footer
        self.status.configure(fg=t["text_fg"], bg=t["window_bg"])
        if self.footer:
            self.footer.configure(fg=t["text_fg"], bg=t["window_bg"])

        # Canvas and Cells
        self.canvas.configure(bg=t["canvas_bg"])
        # Update the Frame to the New Colour
        self._draw_board_frame()
        self._redraw_all()


    # Events / Flow
    def new_game(self):
        self.game.reset()
        self._redraw_all()
        human = self.human_player_var.get()
        self._set_status(f"New game. You are P{human}. Current Player: {self.game.current_player}")

        # If you chose P2, QBot opens
        if human == 2:
            self.after(150, self.qbot_move)


    def on_click(self, event):
        if self.game.finished:
            return

        human = self.human_player_var.get()
        if self.game.current_player != human:
            return  # Not Your Turn

        c = event.x // self.cell
        r = event.y // self.cell
        if not (0 <= r < self.game.rows and 0 <= c < self.game.cols):
            return
        if self.game.board[r][c] == RED:
            self._set_status("That Cell is Already RED! Choose Another One.")
            return

        if not self.game.advance(r, c):
            self._set_status("Invalid Move. Try Another Cell.")
            return

        self._update_cell(r, c)

        if self.game.finished:
            self._end_game_actions()
        else:
            self.after(150, self.qbot_move)


    def qbot_move(self):
        if self.game.finished:
            return

        human = self.human_player_var.get()
        if self.game.current_player == human:
            return  # It is your Turn

        move = self.qbot.choose(self.game)
        if move is None:
            self._set_status("QBot has no Legal Moves.")
            return

        r, c = move
        self.game.advance(r, c)
        self._update_cell(r, c)

        if self.game.finished:
            self._end_game_actions()
        else:
            self._set_status(f"Current Player: {self.game.current_player}")


    def _end_game_actions(self):
        self._highlight_win()
        if self.game.winner is not None:
            col = STATE_TO_NAME.get(self.game.winning_color, "?")
            self._set_status(f"GAME OVER: Player {self.game.winner} Wins with {col} at {self.game.winning_line}")
        else:
            self._set_status("GAME OVER.")




# Launch the UI and Starts the Game
if __name__ == "__main__":
    print("Starting the Game, Please Wait...")
    app = QBotUI(qtable_path="qtable.json")
    app.mainloop()
