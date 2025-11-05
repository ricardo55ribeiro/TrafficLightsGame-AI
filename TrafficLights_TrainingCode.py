from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import random
import json
import os


EMPTY, GREEN, YELLOW, RED = 0, 1, 2, 3
STATE_TO_CHAR = {EMPTY: ".", GREEN: "G", YELLOW: "Y", RED: "R"}


@dataclass
class TrafficLightsGame:
    """Traffic Lights Game Logic"""
    rows: int = 3
    cols: int = 4
    board: List[List[int]] = field(default_factory=lambda: [[EMPTY]*4 for _ in range(3)])
    finished: bool = False
    winning_color: Optional[int] = None
    winning_line: Optional[List[Tuple[int, int]]] = None
    current_player: int = 1
    winner: Optional[int] = None

    def advance(self, r: int, c: int) -> bool:
        if self.finished: return False
        if not (0 <= r < self.rows and 0 <= c < self.cols): return False
        if self.board[r][c] == RED: return False
        self.board[r][c] += 1
        self.current_player = 3 - self.current_player
        won, color, line = self._check_win()
        if won:
            self.finished = True
            self.winning_color = color
            self.winning_line = line
            self.winner = 3 - self.current_player
        return True

    def at(self, r: int, c: int) -> int:
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise ValueError(f"Invalid coordinates: ({r},{c})")
        return self.board[r][c]

    def reset(self) -> None:
        self.board = [[EMPTY for _ in range(self.cols)] for _ in range(self.rows)]
        self.finished = False
        self.winning_color = None
        self.winning_line = None
        self.current_player = 1
        self.winner = None

    def __str__(self) -> str:
        header = "    " + " ".join(str(c) for c in range(self.cols))
        lines = [header, "    " + "-" * (2 * self.cols - 1)]
        for r in range(self.rows):
            row_str = " ".join(STATE_TO_CHAR[self.board[r][c]] for c in range(self.cols))
            lines.append(f"{r} | {row_str}")
        if self.finished and self.winning_color is not None:
            color_char = STATE_TO_CHAR[self.winning_color]
            if self.winner is not None:
                lines.append(f"\nWIN: Player {self.winner} with '{color_char}' at cells {self.winning_line}")
            else:
                lines.append(f"\nWIN: line of three '{color_char}' at cells {self.winning_line}")
        else:
            lines.append(f"\nCurrent Player: {self.current_player}")
        return "\n".join(lines)

    def _check_win(self) -> Tuple[bool, Optional[int], Optional[List[Tuple[int, int]]]]:
        # Horizontals
        for r in range(self.rows):
            for c in (0, 1):
                coords = [(r, c), (r, c + 1), (r, c + 2)]
                color = self._same_non_empty(coords)
                if color is not None: return True, color, coords
        # Verticals
        for c in range(self.cols):
            coords = [(0, c), (1, c), (2, c)]
            color = self._same_non_empty(coords)
            if color is not None: return True, color, coords
        # Diagonals down-right ( \ )
        for c in (0, 1):
            coords = [(0, c), (1, c + 1), (2, c + 2)]
            color = self._same_non_empty(coords)
            if color is not None: return True, color, coords
        # Diagonals down-left ( / )
        for c in (2, 3):
            coords = [(0, c), (1, c - 1), (2, c - 2)]
            color = self._same_non_empty(coords)
            if color is not None: return True, color, coords
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



# Store the data in an alternative file before "merging"
# into the real one, in order to avoid previous training loss
# if the code stops running 

# Atomic write helper
def _atomic_json_write(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

# Append-only delta logger
class DeltaLogger:
    def __init__(self, path: str = "q_deltas.jsonl", flush_every: int = 5000):
        self.path = path
        self.flush_every = flush_every
        self._buf = []
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def append(self, s: str, a: str, v: float):
        self._buf.append({"s": s, "a": a, "v": float(v)})
        if len(self._buf) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self._buf: return
        with open(self.path, "a", encoding="utf-8") as f:
            for rec in self._buf:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._buf.clear()

    def replay_into(self, Q: Dict[str, Dict[str, float]]) -> int:
        if not os.path.exists(self.path): return 0
        applied = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                s, a, v = rec["s"], rec["a"], float(rec["v"])
                Q.setdefault(s, {})[a] = v
                applied += 1
        return applied

    def rotate(self):
        if os.path.exists(self.path):
            os.replace(self.path, self.path + ".done")




# Q-Learning Agent (QBot)
class QBot:
    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 0.99,
        epsilon: float = 0.20,
        seed: Optional[int] = None,
        q_path: Optional[str] = None,
        delta_logger: Optional[DeltaLogger] = None,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self._rng = random.Random(seed)
        self.Q: Dict[str, Dict[str, float]] = {}
        self.q_path = q_path
        self.delta_logger = delta_logger
        if q_path and os.path.exists(q_path):
            self.load(q_path)

    def candidate_moves(self, game: TrafficLightsGame) -> List[Tuple[int, int]]:
        if game.finished: return []
        return [(r, c) for r in range(game.rows) for c in range(game.cols) if game.board[r][c] != RED]

    def choose(self, game: TrafficLightsGame) -> Optional[Tuple[int, int]]:
        moves = self.candidate_moves(game)
        if not moves: return None
        s = self._state_key(game)
        if s not in self.Q: self.Q[s] = {}
        if self._rng.random() < self.epsilon:
            return self._rng.choice(moves)

        # Exploit with survival tie-break
        # (important for positions where the agent realizes the game is already lost
        # if the opponent plays rationality, so he chooses longevity)
        best_val = None
        best_moves = []
        for (r, c) in moves:
            a = self._action_key(r, c)
            val = self.Q[s].get(a, 0.0)
            if (best_val is None) or (val > best_val):
                best_val, best_moves = val, [(r, c)]
            elif val == best_val:
                best_moves.append((r, c))

        if len(best_moves) > 1:
            safe = []
            for (r, c) in best_moves:
                tmp = TrafficLightsGame(rows=game.rows, cols=game.cols)
                tmp.board = [row[:] for row in game.board]
                tmp.current_player = game.current_player
                tmp.advance(r, c)
                opp_has_win = False
                for rr in range(tmp.rows):
                    if opp_has_win: break
                    for cc in range(tmp.cols):
                        if tmp.board[rr][cc] == RED: continue
                        sim = TrafficLightsGame(rows=tmp.rows, cols=tmp.cols)
                        sim.board = [row[:] for row in tmp.board]
                        sim.current_player = tmp.current_player
                        sim.advance(rr, cc)
                        if sim.finished and sim.winner == tmp.current_player:
                            opp_has_win = True; break
                if not opp_has_win: safe.append((r, c))
            if safe:
                return self._rng.choice(safe)
        return self._rng.choice(best_moves)

    def update(self, s: str, a: str, r: float, s_next: Optional[str], legal_next: List[Tuple[int, int]]):
        if s not in self.Q: self.Q[s] = {}
        old = self.Q[s].get(a, 0.0)
        if s_next is None or not legal_next:
            target = r
        else:
            nxt = self.Q.get(s_next, {})
            max_next = max((nxt.get(self._action_key(rn, cn), 0.0) for (rn, cn) in legal_next), default=0.0)
            target = r + self.gamma * max_next
        new_val = old + self.alpha * (target - old)
        self.Q[s][a] = new_val

        if self.delta_logger is not None:
            self.delta_logger.append(s, a, new_val)

    def save(self, path: Optional[str] = None):
        path = path or self.q_path or "qtable.json"
        serial = {s: {a: float(v) for a, v in adict.items()} for s, adict in self.Q.items()}
        _atomic_json_write(path, serial)

    def load(self, path: Optional[str] = None):
        path = path or self.q_path or "qtable.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.Q = {str(s): {str(a): float(v) for a, v in adict.items()} for s, adict in data.items()}

    @staticmethod
    def _state_key(game: TrafficLightsGame) -> str:
        rows = [''.join(str(x) for x in row) for row in game.board]
        return '/'.join(rows) + f'|{game.current_player}'
    
    @staticmethod
    def _action_key(r: int, c: int) -> str:
        return f"{r},{c}"




# Bots used to train Q-Agent
class RandomBot:
    """
    Plays with uniform randomness
    """
    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
    def candidate_moves(self, game: TrafficLightsGame) -> List[Tuple[int, int]]:
        if game.finished: return []
        return [(r, c) for r in range(game.rows) for c in range(game.cols) if game.board[r][c] != RED]
    def choose(self, game: TrafficLightsGame) -> Optional[Tuple[int, int]]:
        moves = self.candidate_moves(game)
        if not moves: return None
        return self._rng.choice(moves)
    def play(self, game: TrafficLightsGame) -> bool:
        mv = self.choose(game)
        if mv is None: return False
        r, c = mv; return game.advance(r, c)


class MyopicBot:
    """
    Checks if it can win in this move, and plays;
    If not, checks if it can lose the next move, to avoid the play,
    and then, randomly plays one of the remaining moves
    """
    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
    def candidate_moves(self, game: TrafficLightsGame) -> List[Tuple[int, int]]:
        if game.finished: return []
        return [(r, c) for r in range(game.rows) for c in range(game.cols) if game.board[r][c] != RED]
    def _would_win_after(self, game: TrafficLightsGame, r: int, c: int) -> bool:
        if game.board[r][c] == RED: return False
        prev = game.board[r][c]; game.board[r][c] = prev + 1
        won, _, _ = game._check_win()
        game.board[r][c] = prev
        return won
    def choose(self, game: TrafficLightsGame) -> Optional[Tuple[int, int]]:
        if game.finished: return None
        moves = self.candidate_moves(game)
        if not moves: return None
        for r, c in moves:
            if self._would_win_after(game, r, c): return (r, c)
        safe = []
        for r, c in moves:
            prev = game.board[r][c]
            game.board[r][c] = prev + 1
            opp_win = False
            for rr in range(game.rows):
                if opp_win: break
                for cc in range(game.cols):
                    if game.board[rr][cc] == RED: continue
                    opp_prev = game.board[rr][cc]
                    game.board[rr][cc] = opp_prev + 1
                    won, _, _ = game._check_win()
                    game.board[rr][cc] = opp_prev
                    if won: opp_win = True; break
            game.board[r][c] = prev
            if not opp_win: safe.append((r, c))
        if safe: return self._rng.choice(safe)
        return self._rng.choice(moves)
    def play(self, game: TrafficLightsGame) -> bool:
        mv = self.choose(game)
        if mv is None: return False
        r, c = mv; return game.advance(r, c)


class AlternateBot:
    """
    MyopicBot for the first [10,20] moves (the exact number of moves is randomly decided for each game, 
    inside this interval), then RandomBot
    """
    def __init__(self, switch_min_ply: int = 10, switch_max_ply: int = 20, seed: Optional[int] = None):
        self.myopic = MyopicBot(seed=seed)
        self.random = RandomBot(seed=seed)
        self.rng = random.Random(seed)
        self.switch_min_ply = switch_min_ply
        self.switch_max_ply = switch_max_ply
        self.switch_at = switch_min_ply
        self.ply = 0
    def new_game(self):
        self.switch_at = self.rng.randint(self.switch_min_ply, self.switch_max_ply)
        self.ply = 0
    def choose(self, game: TrafficLightsGame) -> Optional[Tuple[int, int]]:
        if game.finished: return None
        if self.ply < self.switch_at:
            return self.myopic.choose(game)
        else:
            return self.random.choose(game)
    def on_move_played(self):
        self.ply += 1



# Functions used for training

def train_qbot_vs_randombot(
    n_games: int,
    q_path: str = "qtable.json",
    alpha: float = 0.5,
    gamma: float = 0.99,
    epsilon_start: float = 0.20,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.9995,
    save_every: int = 5000,
    verbose_every: int = 1000,
    seed: Optional[int] = None,
    checkpoint_path: str = "random_checkpoint.json",
    resume: bool = True,
    delta_path: str = "q_deltas.jsonl",
):
    delta = DeltaLogger(path=delta_path, flush_every=5000)
    qb = QBot(alpha=alpha, gamma=gamma, epsilon=epsilon_start, seed=seed,
              q_path=q_path if os.path.exists(q_path) else None,
              delta_logger=delta)
    opp = RandomBot()
    delta.replay_into(qb.Q)

    start_game = 1
    eps = epsilon_start
    if resume and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                ck = json.load(f)
            start_game = int(ck.get("next_game", 1))
            eps = float(ck.get("epsilon", epsilon_start))
            eps = max(epsilon_end, min(eps, epsilon_start))
            print(f"[resume] starting at game {start_game} with ε={eps:.4f}")
        except Exception:
            pass
    qb.epsilon = eps

    def save_checkpoint(next_game: int, epsilon_val: float):
        _atomic_json_write(checkpoint_path, {
            "next_game": next_game,
            "epsilon": float(epsilon_val),
            "qtable_rows": len(qb.Q),
        })

    qb_wins = rnd_wins = 0
    p1_wins = p2_wins = 0

    try:
        for g_idx in range(start_game, start_game + n_games):
            qb.epsilon = eps if g_idx == start_game else max(epsilon_end, qb.epsilon * epsilon_decay)
            eps = qb.epsilon

            game = TrafficLightsGame()
            qb_player = 1 if (g_idx % 2 == 1) else 2
            pending_s = None
            pending_a = None

            while not game.finished:
                if game.current_player == qb_player:
                    s = qb._state_key(game)
                    move = qb.choose(game)
                    if move is None:
                        if pending_s is not None and pending_a is not None:
                            qb.update(pending_s, pending_a, 0.0, None, [])
                            pending_s = pending_a = None
                        break
                    r, c = move
                    a_key = qb._action_key(r, c)
                    ok = game.advance(r, c)
                    if not ok:
                        qb.update(s, a_key, -1.0, None, [])
                        break
                    if game.finished:
                        qb.update(s, a_key, +1.0, None, [])
                        qb_wins += 1
                        if game.winner == 1: p1_wins += 1
                        else: p2_wins += 1
                        break
                    pending_s, pending_a = s, a_key
                else:
                    move_opp = opp.choose(game)
                    if move_opp is None:
                        if pending_s is not None and pending_a is not None:
                            qb.update(pending_s, pending_a, 0.0, None, [])
                            pending_s = pending_a = None
                        break
                    rr, cc = move_opp
                    ok = game.advance(rr, cc)
                    if not ok:
                        if pending_s is not None and pending_a is not None:
                            qb.update(pending_s, pending_a, 0.0, None, [])
                            pending_s = pending_a = None
                        break
                    if game.finished:
                        if pending_s is not None and pending_a is not None:
                            qb.update(pending_s, pending_a, -1.0, None, [])
                            pending_s = pending_a = None
                        rnd_wins += 1
                        if game.winner == 1: p1_wins += 1
                        else: p2_wins += 1
                        break
                    else:
                        if pending_s is not None and pending_a is not None:
                            s_next = qb._state_key(game)
                            legal_next = qb.candidate_moves(game)
                            qb.update(pending_s, pending_a, 0.0, s_next, legal_next)
                            pending_s = pending_a = None

            if save_every and (g_idx % save_every == 0):
                delta.flush()
                save_checkpoint(g_idx + 1, eps)

            if verbose_every and (g_idx % verbose_every == 0):
                total = qb_wins + rnd_wins
                print(f"[{g_idx}] ε={qb.epsilon:.4f}  QBot:{qb_wins}  Random:{rnd_wins}  "
                      f"(P1:{p1_wins} P2:{p2_wins}, decided={total})")

        # Consolidate once
        delta.flush()
        if os.path.exists(q_path):
            with open(q_path, "r", encoding="utf-8") as f:
                bigQ = json.load(f)
        else:
            bigQ = {}
        _ = delta.replay_into(bigQ)
        _atomic_json_write(q_path, bigQ)
        delta.rotate()
        save_checkpoint(g_idx + 1, eps)

    except KeyboardInterrupt:
        print("\n[train] Ctrl+C. Flushing deltas and consolidating atomically...")
        delta.flush()
        if os.path.exists(q_path):
            with open(q_path, "r", encoding="utf-8") as f:
                bigQ = json.load(f)
        else:
            bigQ = {}
        _ = delta.replay_into(bigQ)
        _atomic_json_write(q_path, bigQ)
        delta.rotate()
        next_g = (g_idx + 1) if 'g_idx' in locals() else start_game
        save_checkpoint(next_g, eps)

    return qb, {"QBot": qb_wins, "Random": rnd_wins, "P1": p1_wins, "P2": p2_wins}


def train_qbot_vs_myopicbot(
    n_games: int,
    q_path: str = "qtable.json",
    alpha: float = 0.5,
    gamma: float = 0.99,
    epsilon_start: float = 0.20,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.9995,
    save_every: int = 5000,
    verbose_every: int = 1000,
    seed: Optional[int] = None,
    checkpoint_path: str = "myopic_checkpoint.json",
    resume: bool = True,
    delta_path: str = "q_deltas.jsonl",
):
    delta = DeltaLogger(path=delta_path, flush_every=5000)
    qb = QBot(alpha=alpha, gamma=gamma, epsilon=epsilon_start, seed=seed,
              q_path=q_path if os.path.exists(q_path) else None,
              delta_logger=delta)
    opp = MyopicBot()
    delta.replay_into(qb.Q)

    start_game = 1
    eps = epsilon_start
    if resume and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                ck = json.load(f)
            start_game = int(ck.get("next_game", 1))
            eps = float(ck.get("epsilon", epsilon_start))
            eps = max(epsilon_end, min(eps, epsilon_start))
            print(f"[resume] starting at game {start_game} with ε={eps:.4f}")
        except Exception:
            pass
    qb.epsilon = eps

    def save_checkpoint(next_game: int, epsilon_val: float):
        _atomic_json_write(checkpoint_path, {
            "next_game": next_game,
            "epsilon": float(epsilon_val),
            "qtable_rows": len(qb.Q),
        })

    qb_wins = opp_wins = 0
    p1_wins = p2_wins = 0

    try:
        for g_idx in range(start_game, start_game + n_games):
            qb.epsilon = eps if g_idx == start_game else max(epsilon_end, qb.epsilon * epsilon_decay)
            eps = qb.epsilon

            game = TrafficLightsGame()
            qb_player = 1 if (g_idx % 2 == 1) else 2
            pending_s = None
            pending_a = None

            while not game.finished:
                if game.current_player == qb_player:
                    s = qb._state_key(game)
                    move = qb.choose(game)
                    if move is None:
                        if pending_s is not None and pending_a is not None:
                            qb.update(pending_s, pending_a, 0.0, None, [])
                            pending_s = pending_a = None
                        break
                    r, c = move
                    a_key = qb._action_key(r, c)
                    ok = game.advance(r, c)
                    if not ok:
                        qb.update(s, a_key, -1.0, None, [])
                        break
                    if game.finished:
                        qb.update(s, a_key, +1.0, None, [])
                        qb_wins += 1
                        if game.winner == 1: p1_wins += 1
                        else: p2_wins += 1
                        break
                    pending_s, pending_a = s, a_key
                else:
                    move_opp = opp.choose(game)
                    if move_opp is None:
                        if pending_s is not None and pending_a is not None:
                            qb.update(pending_s, pending_a, 0.0, None, [])
                            pending_s = pending_a = None
                        break
                    rr, cc = move_opp
                    ok = game.advance(rr, cc)
                    if not ok:
                        if pending_s is not None and pending_a is not None:
                            qb.update(pending_s, pending_a, 0.0, None, [])
                            pending_s = pending_a = None
                        break
                    if game.finished:
                        if pending_s is not None and pending_a is not None:
                            qb.update(pending_s, pending_a, -1.0, None, [])
                            pending_s = pending_a = None
                        opp_wins += 1
                        if game.winner == 1: p1_wins += 1
                        else: p2_wins += 1
                        break
                    else:
                        if pending_s is not None and pending_a is not None:
                            s_next = qb._state_key(game)
                            legal_next = qb.candidate_moves(game)
                            qb.update(pending_s, pending_a, 0.0, s_next, legal_next)
                            pending_s = pending_a = None

            if save_every and (g_idx % save_every == 0):
                delta.flush()
                save_checkpoint(g_idx + 1, eps)

            if verbose_every and (g_idx % verbose_every == 0):
                total = qb_wins + opp_wins
                print(f"[{g_idx}] ε={qb.epsilon:.4f}  QBot:{qb_wins}  Myopic:{opp_wins}  "
                      f"(P1:{p1_wins} P2:{p2_wins}, decided={total})")

        delta.flush()
        if os.path.exists(q_path):
            with open(q_path, "r", encoding="utf-8") as f:
                bigQ = json.load(f)
        else:
            bigQ = {}
        _ = delta.replay_into(bigQ)
        _atomic_json_write(q_path, bigQ)
        delta.rotate()
        save_checkpoint(g_idx + 1, eps)

    except KeyboardInterrupt:
        print("\n[train] Ctrl+C. Flushing deltas and consolidating atomically...")
        delta.flush()
        if os.path.exists(q_path):
            with open(q_path, "r", encoding="utf-8") as f:
                bigQ = json.load(f)
        else:
            bigQ = {}
        _ = delta.replay_into(bigQ)
        _atomic_json_write(q_path, bigQ)
        delta.rotate()
        next_g = (g_idx + 1) if 'g_idx' in locals() else start_game
        save_checkpoint(next_g, eps)

    return qb, {"QBot": qb_wins, "Myopic": opp_wins, "P1": p1_wins, "P2": p2_wins}


def train_qbot_vs_alternatebot(
    n_games: int,
    switch_min_ply: int = 10,
    switch_max_ply: int = 20,
    q_path: str = "qtable.json",
    alpha: float = 0.5,
    gamma: float = 0.99,
    epsilon_start: float = 0.20,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.9995,
    save_every: int = 5000,
    verbose_every: int = 2000,
    seed: Optional[int] = None,
    checkpoint_path: str = "alternate_checkpoint.json",
    resume: bool = True,
    delta_path: str = "q_alternate_deltas.jsonl",
):
    delta = DeltaLogger(path=delta_path, flush_every=5000)
    qb = QBot(alpha=alpha, gamma=gamma, epsilon=epsilon_start,
              seed=seed,
              q_path=q_path if os.path.exists(q_path) else None,
              delta_logger=delta)
    opp = AlternateBot(switch_min_ply=switch_min_ply, switch_max_ply=switch_max_ply, seed=seed)

    delta.replay_into(qb.Q)

    start_game = 1
    eps = epsilon_start
    if resume and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                ck = json.load(f)
            start_game = int(ck.get("next_game", 1))
            eps = float(ck.get("epsilon", epsilon_start))
            eps = max(epsilon_end, min(eps, epsilon_start))
            print(f"[resume] starting at game {start_game} with ε={eps:.4f}")
        except Exception:
            pass
    qb.epsilon = eps

    def save_checkpoint(next_game: int, epsilon_val: float):
        _atomic_json_write(checkpoint_path, {
            "next_game": next_game,
            "epsilon": float(epsilon_val),
            "qtable_rows": len(qb.Q),
        })

    def consolidate_once():
        delta.flush()
        if os.path.exists(q_path):
            with open(q_path, "r", encoding="utf-8") as f:
                bigQ = json.load(f)
        else:
            bigQ = {}
        _ = delta.replay_into(bigQ)
        _atomic_json_write(q_path, bigQ)
        delta.rotate()

    qb_wins = opp_wins = 0
    p1_wins = p2_wins = 0

    try:
        for g_idx in range(start_game, start_game + n_games):
            qb.epsilon = eps if g_idx == start_game else max(epsilon_end, qb.epsilon * epsilon_decay)
            eps = qb.epsilon

            opp.new_game()
            game = TrafficLightsGame()
            qb_player = 1 if (g_idx % 2 == 1) else 2

            pending_s = None
            pending_a = None

            while not game.finished:
                if game.current_player == qb_player:
                    s = qb._state_key(game)
                    move = qb.choose(game)
                    if move is None:
                        if pending_s is not None and pending_a is not None:
                            qb.update(pending_s, pending_a, 0.0, None, [])
                            pending_s = pending_a = None
                        break
                    r, c = move
                    a_key = qb._action_key(r, c)
                    ok = game.advance(r, c)
                    if not ok:
                        qb.update(s, a_key, -1.0, None, [])
                        break
                    if game.finished:
                        qb.update(s, a_key, +1.0, None, [])
                        qb_wins += 1
                        if game.winner == 1: p1_wins += 1
                        else: p2_wins += 1
                        break
                    pending_s, pending_a = s, a_key
                else:
                    move_opp = opp.choose(game)
                    if move_opp is None:
                        if pending_s is not None and pending_a is not None:
                            qb.update(pending_s, pending_a, 0.0, None, [])
                            pending_s = pending_a = None
                        break
                    rr, cc = move_opp
                    ok = game.advance(rr, cc)
                    if ok:
                        opp.on_move_played()
                    if not ok:
                        if pending_s is not None and pending_a is not None:
                            qb.update(pending_s, pending_a, 0.0, None, [])
                            pending_s = pending_a = None
                        break
                    if game.finished:
                        if pending_s is not None and pending_a is not None:
                            qb.update(pending_s, pending_a, -1.0, None, [])
                            pending_s = pending_a = None
                        opp_wins += 1
                        if game.winner == 1: p1_wins += 1
                        else: p2_wins += 1
                        break
                    else:
                        if pending_s is not None and pending_a is not None:
                            s_next = qb._state_key(game)
                            legal_next = qb.candidate_moves(game)
                            qb.update(pending_s, pending_a, 0.0, s_next, legal_next)
                            pending_s = pending_a = None

            if save_every and (g_idx % save_every == 0):
                consolidate_once()
                save_checkpoint(g_idx + 1, eps)

            if verbose_every and (g_idx % verbose_every == 0):
                decided = qb_wins + opp_wins
                print(f"[{g_idx}] ε={eps:.4f}  QBot:{qb_wins}  Alternate:{opp_wins}  "
                      f"(P1:{p1_wins} P2:{p2_wins}, decided={decided})")

        consolidate_once()
        save_checkpoint(g_idx + 1, eps)

    except KeyboardInterrupt:
        print("\n[train] Ctrl+C. Flushing deltas and consolidating atomically...")
        consolidate_once()
        next_g = (g_idx + 1) if 'g_idx' in locals() else start_game
        save_checkpoint(next_g, eps)

    return qb, {"QBot": qb_wins, "Alternate": opp_wins, "P1": p1_wins, "P2": p2_wins}


# QBot vs QBot training function, the Agent plays against himself
def train_qbot_vs_qbot_shared(
    n_games: int,
    q_path: str = "qtable.json",
    alpha: float = 0.5,
    gamma: float = 0.99,
    epsilon_start: float = 0.20,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.9995,
    save_every: int = 2000,
    verbose_every: int = 2000,
    seed: Optional[int] = None,
    checkpoint_path: str = "selfplay_checkpoint.json",
    resume: bool = True,
):
    delta = DeltaLogger(path="qtable_deltas.jsonl", flush_every=5000)

    base = QBot(alpha=alpha, gamma=gamma, epsilon=epsilon_start,
                seed=seed,
                q_path=q_path if os.path.exists(q_path) else None,
                delta_logger=delta)
    qb1 = base
    qb2_seed = None if seed is None else seed + 1
    qb2 = QBot(alpha=alpha, gamma=gamma, epsilon=epsilon_start,
               seed=qb2_seed,
               delta_logger=delta)
    qb2.Q = qb1.Q
    qb2.q_path = q_path

    delta.replay_into(qb1.Q)

    start_game = 1
    eps = epsilon_start
    if resume and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                ck = json.load(f)
            start_game = int(ck.get("next_game", 1))
            eps = float(ck.get("epsilon", epsilon_start))
            eps = max(epsilon_end, min(eps, epsilon_start))
            print(f"[resume] starting at game {start_game} with ε={eps:.4f}")
        except Exception:
            pass

    qb1_wins = qb2_wins = 0
    p1_wins = p2_wins = 0

    def save_checkpoint(next_game: int, epsilon_val: float):
        _atomic_json_write(checkpoint_path, {
            "next_game": next_game,
            "epsilon": float(epsilon_val),
            "qtable_rows": len(qb1.Q),
        })

    try:
        for g_idx in range(start_game, start_game + n_games):
            if g_idx == start_game:
                qb1.epsilon = qb2.epsilon = eps
            else:
                eps = max(epsilon_end, eps * epsilon_decay)
                qb1.epsilon = qb2.epsilon = eps

            game = TrafficLightsGame()
            qb1_seat = 1 if (g_idx % 2 == 1) else 2
            qb2_seat = 3 - qb1_seat

            p_s_1 = p_a_1 = None
            p_s_2 = p_a_2 = None

            while not game.finished:
                if game.current_player == qb1_seat:
                    agent = qb1
                    opp_pending = (2, p_s_2, p_a_2)
                    set_pending = 1
                else:
                    agent = qb2
                    opp_pending = (1, p_s_1, p_a_1)
                    set_pending = 2

                s = agent._state_key(game)
                move = agent.choose(game)
                if move is None:
                    k, ps, pa = opp_pending
                    if ps is not None and pa is not None:
                        agent.update(ps, pa, 0.0, None, [])
                        if k == 1: p_s_1 = p_a_1 = None
                        else:      p_s_2 = p_a_2 = None
                    break

                r, c = move
                a_key = agent._action_key(r, c)
                ok = game.advance(r, c)
                if not ok:
                    agent.update(s, a_key, -1.0, None, [])
                    break

                if game.finished:
                    agent.update(s, a_key, +1.0, None, [])
                    k, ps, pa = opp_pending
                    if ps is not None and pa is not None:
                        agent.update(ps, pa, -1.0, None, [])
                        if k == 1: p_s_1 = p_a_1 = None
                        else:      p_s_2 = p_a_2 = None

                    if game.winner == 1: p1_wins += 1
                    else:                 p2_wins += 1
                    if (game.winner == qb1_seat): qb1_wins += 1
                    else:                         qb2_wins += 1
                    break
                else:
                    k, ps, pa = opp_pending
                    if ps is not None and pa is not None:
                        s_next = agent._state_key(game)
                        legal_next = agent.candidate_moves(game)
                        agent.update(ps, pa, 0.0, s_next, legal_next)
                        if k == 1: p_s_1 = p_a_1 = None
                        else:      p_s_2 = p_a_2 = None

                    if set_pending == 1:
                        p_s_1, p_a_1 = s, a_key
                    else:
                        p_s_2, p_a_2 = s, a_key

            if save_every and (g_idx % save_every == 0):
                delta.flush()
                save_checkpoint(g_idx + 1, eps)

            if verbose_every and (g_idx % verbose_every == 0):
                decided = qb1_wins + qb2_wins
                print(f"[{g_idx}] ε={eps:.4f}  qb1:{qb1_wins} qb2:{qb2_wins}  "
                      f"(P1:{p1_wins} P2:{p2_wins}, decided={decided})")

        delta.flush()
        if os.path.exists(q_path):
            with open(q_path, "r", encoding="utf-8") as f:
                bigQ = json.load(f)
        else:
            bigQ = {}
        _ = delta.replay_into(bigQ)
        _atomic_json_write(q_path, bigQ)
        delta.rotate()
        save_checkpoint(g_idx + 1, eps)

    except KeyboardInterrupt:
        print("\n[train] Ctrl+C detected. Flushing deltas and consolidating once...")
        delta.flush()
        if os.path.exists(q_path):
            with open(q_path, "r", encoding="utf-8") as f:
                bigQ = json.load(f)
        else:
            bigQ = {}
        _ = delta.replay_into(bigQ)
        _atomic_json_write(q_path, bigQ)
        delta.rotate()
        next_g = (g_idx + 1) if 'g_idx' in locals() else start_game
        save_checkpoint(next_g, eps)

    return qb1, {"qb1": qb1_wins, "qb2": qb2_wins, "P1": p1_wins, "P2": p2_wins}



# Functions to evaluate QBot
def eval_qbot_vs_random(n_games: int = 1000, q_path: str = "qtable.json", seed: Optional[int] = None):
    qb = QBot(alpha=0.0, gamma=0.99, epsilon=0.0, seed=seed, q_path=q_path)
    opp = RandomBot()
    qb_wins = opp_wins = 0
    p1_wins = p2_wins = 0
    for i in range(1, n_games + 1):
        game = TrafficLightsGame()
        qb_player = 1 if (i % 2 == 1) else 2
        while not game.finished:
            bot = qb if game.current_player == qb_player else opp
            move = bot.choose(game)
            if move is None: break
            r, c = move
            game.advance(r, c)
        if game.winner == qb_player: qb_wins += 1
        elif game.winner is not None: opp_wins += 1
        if game.winner == 1: p1_wins += 1
        elif game.winner == 2: p2_wins += 1
    return {"QBot": qb_wins, "Random": opp_wins, "P1": p1_wins, "P2": p2_wins}

def eval_qbot_vs_myopic(n_games: int = 1000, q_path: str = "qtable.json", seed: Optional[int] = None):
    qb = QBot(alpha=0.0, gamma=0.99, epsilon=0.0, seed=seed, q_path=q_path)
    opp = MyopicBot()
    qb_wins = opp_wins = 0
    p1_wins = p2_wins = 0
    for i in range(1, n_games + 1):
        game = TrafficLightsGame()
        qb_player = 1 if (i % 2 == 1) else 2
        while not game.finished:
            bot = qb if game.current_player == qb_player else opp
            move = bot.choose(game)
            if move is None: break
            r, c = move
            game.advance(r, c)
        if game.winner == qb_player: qb_wins += 1
        elif game.winner is not None: opp_wins += 1
        if game.winner == 1: p1_wins += 1
        elif game.winner == 2: p2_wins += 1
    return {"QBot": qb_wins, "Myopic": opp_wins, "P1": p1_wins, "P2": p2_wins}

def eval_qbot_vs_alternate(
    n_games: int = 1000,
    switch_min_ply: int = 10,
    switch_max_ply: int = 20,
    q_path: str = "qtable.json",
    seed: Optional[int] = None,
):
    qb = QBot(alpha=0.0, gamma=0.99, epsilon=0.0, seed=seed, q_path=q_path)
    opp = AlternateBot(switch_min_ply=switch_min_ply, switch_max_ply=switch_max_ply, seed=seed)

    qb_wins = opp_wins = 0
    p1_wins = p2_wins = 0

    for i in range(1, n_games + 1):
        game = TrafficLightsGame()
        qb_player = 1 if (i % 2 == 1) else 2
        opp.new_game()

        while not game.finished:
            if game.current_player == qb_player:
                move = qb.choose(game)
                if move is None: break
                r, c = move
                game.advance(r, c)
            else:
                move = opp.choose(game)
                if move is None: break
                rr, cc = move
                ok = game.advance(rr, cc)
                if ok: opp.on_move_played()

        if game.winner == qb_player: qb_wins += 1
        elif game.winner is not None: opp_wins += 1
        if game.winner == 1: p1_wins += 1
        elif game.winner == 2: p2_wins += 1

    return {"QBot": qb_wins, "Alternate": opp_wins, "P1": p1_wins, "P2": p2_wins}



# CLI helpers
def _read_positive_int(prompt: str) -> int:
    while True:
        try:
            n = int(input(prompt).strip())
            if n > 0: return n
        except: pass
        print("Please enter a positive integer (e.g., 10000).")



if __name__ == "__main__":
    choice = input("With what bot do you want to train: Random, Alternate, Myopic or QBot? (R, A, M, Q) \nOr do you want to evaluate? (1 for R, 2 for A, 3 for M) ").strip().upper()
    choice = choice[:1] if choice else ""

    if choice == "R":
        n = _read_positive_int("How many training games (QBot vs RandomBot)? ")
        bot, stats = train_qbot_vs_randombot(
            n_games=n,
            q_path="qtable.json",
            alpha=0.5, gamma=0.99,
            epsilon_start=0.2, epsilon_end=0.005, epsilon_decay=0.9995,
            save_every=5000, verbose_every=2000,
            checkpoint_path="random_checkpoint.json",
            resume=True,
            delta_path="q_deltas.jsonl",
        )
        print("Training finished. Agent win counts:", stats)

    elif choice == "M":
        n = _read_positive_int("How many training games (QBot vs MyopicBot)? ")
        bot, stats = train_qbot_vs_myopicbot(
            n_games=n,
            q_path="qtable.json",
            alpha=0.5, gamma=0.99,
            epsilon_start=0.2, epsilon_end=0.005, epsilon_decay=0.9995,
            save_every=5000, verbose_every=2000,
            checkpoint_path="myopic_checkpoint.json",
            resume=True,
            delta_path="q_deltas.jsonl",
        )
        print("Training finished. Agent win counts:", stats)

    elif choice == "A":
        n = _read_positive_int("How many training games (QBot vs AlternateBot)? ")
        bot, stats = train_qbot_vs_alternatebot(
            n_games=n,
            switch_min_ply=12, switch_max_ply=18,
            q_path="qtable.json",
            alpha=0.5, gamma=0.99,
            epsilon_start=0.2, epsilon_end=0.01, epsilon_decay=0.9995,
            save_every=5000, verbose_every=2000,
            checkpoint_path="alternate_checkpoint.json",
            resume=True,
            delta_path="q_alternate_deltas.jsonl",
        )
        print("Training finished. Agent/seat wins:", stats)

    elif choice == "Q":
        n = _read_positive_int("How many training games (QBot vs QBot self-play)? ")
        bot, stats = train_qbot_vs_qbot_shared(
            n_games=n,
            q_path="qtable.json",
            alpha=0.5, gamma=0.99,
            epsilon_start=0.05, epsilon_end=0.005, epsilon_decay=0.9995,
            save_every=5000, verbose_every=5000,
            checkpoint_path="selfplay_checkpoint.json",
            resume=True,
        )
        print("Self-play finished. Agent/seat wins:", stats)

    elif choice == "1":
        n = _read_positive_int("How many evaluation games (QBot vs RandomBot)? ")
        eval_stats = eval_qbot_vs_random(n_games=n, q_path="qtable.json")
        print("Evaluation vs RandomBot:", eval_stats)
    
    elif choice == "2":
        n = _read_positive_int("How many evaluation games (QBot vs AlternateBot)? ")
        eval_stats = eval_qbot_vs_alternate(n_games=n, q_path="qtable.json")
        print("Evaluation vs AlternateBot:", eval_stats)
    
    elif choice == "3":
        n = _read_positive_int("How many evaluation games (QBot vs MyopicBot)? ")
        eval_stats = eval_qbot_vs_myopic(n_games=n, q_path="qtable.json")
        print("Evaluation vs MyopicBot:", eval_stats)
    
    else:
        print("Invalid choice. Please run again and pick R, A, M, or Q to train the agent; \nOr in alternative, pick 1, 2 or 3 to evaluate it.\n")
        raise SystemExit(1)

    if choice != "1" and choice != "2" and choice != "3":
        print("States learned:", len(bot.Q))
        print("Total (state,action) entries:", sum(len(a) for a in bot.Q.values()))
