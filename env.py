import subprocess, numpy as np, gymnasium as gym
from gymnasium import spaces
import pathlib

class LanderProcEnv(gym.Env):
    def __init__(self, exe_path:str, obs_dim:int=8, act_dim:int=3):
        super().__init__()
        self.exe_path = exe_path
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)
        self.p = None

    def _start(self):
        if self.p is None or self.p.poll() is not None:
            self.p = subprocess.Popen(
                [self.exe_path],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                universal_newlines=True, bufsize=1
            )

    def _read_obs_line(self):
        line = self.p.stdout.readline().strip()
        if not line.startswith("obs"):
            raise RuntimeError(f"bad line from sim: {line}")
        parts = [s.strip() for s in line.split('|')]
        obs = np.fromstring(parts[0][3:].strip(), sep=' ', dtype=np.float32)
        r = float(parts[1].split()[1]) if len(parts) >= 2 else 0.0
        d = bool(int(parts[2].split()[1])) if len(parts) >= 3 else False
        return obs, r, d

    def reset(self, seed=None, options=None):
        self._start()
        seed = int(0 if seed is None else seed)
        self.p.stdin.write(f"reset {seed}\n"); self.p.stdin.flush()
        obs, _, _ = self._read_obs_line()
        return obs, {}

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)
        self.p.stdin.write("step " + " ".join(f"{x:.6f}" for x in a) + "\n"); self.p.stdin.flush()
        obs, r, d = self._read_obs_line()
        return obs, float(r), bool(d), False, {}

    def close(self):
        if self.p and self.p.poll() is None:
            try:
                self.p.stdin.write("close\n"); self.p.stdin.flush()
            except Exception:
                pass
            self.p.terminate()
            self.p = None
