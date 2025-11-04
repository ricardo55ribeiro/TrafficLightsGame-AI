from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
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

    def advance(self, r: int, c: int) -> bool:
        """Advance (r,c) by exactly one state. Return True on success."""
        if self.finished:
            return False

        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False

        if self.board[r][c] == RED:
            return False

        # Make the move
        self.board[r][c] += 1

        # Switch player after a successful move
        self.current_player = 3 - self.current_player

        # Win check
        won, color, line = self._check_win()
        if won:
            self.finished = True
            self.winning_color = color
            self.winning_line = line
            # Since we already switched, winner is the previous player
            self.winner = 3 - self.current_player

        return True

    # Utility
    def at(self, r: int, c: int) -> int:
        self._validate_coords(r, c)
        return self.board[r][c]

    def reset(self) -> None:
        self.board = [[EMPTY for _ in range(self.cols)] for _ in range(self.rows)]
        self.finished = False
        self.winning_color = None
        self.winning_line = None
        self.current_player = 1
        self.winner = None

    # Internal helpers
    def _validate_coords(self, r: int, c: int) -> None:
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise ValueError(f"Invalid coordinates: ({r}, {c}) out of bounds.")

    def _check_win(self) -> Tuple[bool, Optional[int], Optional[List[Tuple[int, int]]]]:
        """Return (won, color, line_coords)."""
        # Horizontal segments
        for r in range(self.rows):
            for c in (0, 1):
                coords = [(r, c), (r, c + 1), (r, c + 2)]
                color = self._same_non_empty(coords)
                if color is not None:
                    return True, color, coords

        # Vertical segments
        for c in range(self.cols):
            coords = [(0, c), (1, c), (2, c)]
            color = self._same_non_empty(coords)
            if color is not None:
                return True, color, coords

        # Diagonals (down-right: \)
        for c in (0, 1):
            coords = [(0, c), (1, c + 1), (2, c + 2)]
            color = self._same_non_empty(coords)
            if color is not None:
                return True, color, coords

        # Diagonals (down-left: /)
        for c in (2, 3):
            coords = [(0, c), (1, c - 1), (2, c - 2)]
            color = self._same_non_empty(coords)
            if color is not None:
                return True, color, coords

        return False, None, None

    def _same_non_empty(self, coords: List[Tuple[int, int]]) -> Optional[int]:
        r0, c0 = coords[0]
        color = self.board[r0][c0]
        if color == EMPTY:
            return None
        for (r, c) in coords[1:]:
            if self.board[r][c] != color:
                return None
        return color


# Tkinter UI
STATE_COLORS = {
    EMPTY:  "#ffffff",
    GREEN:  "#2ecc71",
    YELLOW: "#f1c40f",
    RED:    "#e74c3c",
}

STATE_TO_NAME = {EMPTY: "Empty", GREEN: "Green", YELLOW: "Yellow", RED: "Red"}

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


class TwoPlayerUI(tk.Tk):
    """
    Player vs Player UI
    """
    def __init__(self, cell_size: int = 200):
        super().__init__()
        self.title("Traffic Lights Game")
        self.resizable(True, True)

        # Engine
        self.game = TrafficLightsGame()

        self.cell = cell_size
        self.rect_ids = [[None]*self.game.cols for _ in range(self.game.rows)]

        # Theme
        self.theme = "day"
        self.board_frame_ids = None

        # Top Controls Bar
        self.bar = tk.Frame(self, padx=10, pady=8, bg=THEMES[self.theme]["bar_bg"])
        self.bar.pack(fill="x")

        # New Game button
        self.newgame_btn = tk.Button(self.bar, text="New Game", command=self.new_game)
        self.newgame_btn.pack(side="left", padx=(6, 6))

        # Theme toggle button
        self.theme_btn = tk.Button(self.bar, text="🌞", command=self.toggle_theme, bd=0, highlightthickness=0)
        self.theme_btn.pack(side="right")

        # Status
        self.status_var = tk.StringVar()
        self.status = tk.Label(self, textvariable=self.status_var)
        self.status.pack(padx=10, pady=(0, 8), anchor="w")

        # Board Canvas
        w = self.game.cols * self.cell
        h = self.game.rows * self.cell
        self.canvas = tk.Canvas(self, width=w, height=h, bg=THEMES[self.theme]["canvas_bg"], highlightthickness=0)
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_click)

        # Layout sizing
        EXTRA_W = 240
        TOPBAR_PAD = 150
        self.geometry(f"{w + EXTRA_W}x{h + TOPBAR_PAD}")
        try:
            self.state('zoomed')
        except tk.TclError:
            pass

        # Draw
        self._init_cells()
        self._set_status("Press New Game to start. Player 1 moves first.")

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
        """Extra right/bottom frame lines so the outer border is visible."""
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

    # Theme helpers
    def _outline_color(self):
        return THEMES[self.theme]["outline"]

    def _cell_fill(self, state: int):
        if self.theme == "night" and state == EMPTY:
            return "#000000"
        return STATE_COLORS[state]

    def toggle_theme(self):
        self.theme = "night" if self.theme == "day" else "day"
        self.theme_btn.config(text="🌙" if self.theme == "night" else "🌞")
        self.apply_theme()

    def apply_theme(self):
        t = THEMES[self.theme]

        # Window + top bar
        self.configure(bg=t["window_bg"])
        self.bar.configure(bg=t["bar_bg"])

        # Top controls
        for w in (self.newgame_btn, self.theme_btn):
            try:
                w.configure(bg=t["bar_bg"], fg=t["text_fg"], activeforeground=t["text_fg"])
            except tk.TclError:
                try:
                    w.configure(bg=t["bar_bg"], fg=t["text_fg"])
                except tk.TclError:
                    pass

        # Canvas and cells
        self.canvas.configure(bg=t["canvas_bg"])
        self._redraw_all()

    # Events / Flow
    def new_game(self):
        self.game.reset()
        self._redraw_all()
        self._set_status(f"New game. Current Player: {self.game.current_player}")

    def on_click(self, event):
        if self.game.finished:
            return

        c = event.x // self.cell
        r = event.y // self.cell
        if not (0 <= r < self.game.rows and 0 <= c < self.game.cols):
            return

        # Feedback if clicking a RED cell
        if self.game.board[r][c] == RED:
            self._set_status("That cell is already RED! Choose another one.")
            return

        if not self.game.advance(r, c):
            self._set_status("Invalid move. Try another cell.")
            return

        self._update_cell(r, c)

        if self.game.finished:
            self._end_game_actions()
        else:
            self._set_status(f"Current Player: {self.game.current_player}")

    def _end_game_actions(self):
        self._highlight_win()
        if self.game.winner is not None:
            col = STATE_TO_NAME.get(self.game.winning_color, "?")
            self._set_status(f"GAME OVER: Player {self.game.winner} wins with {col} at {self.game.winning_line}")
        else:
            self._set_status("GAME OVER.")



if __name__ == "__main__":
    print("Starting the Game...")
    app = TwoPlayerUI(cell_size=200)
    app.mainloop()