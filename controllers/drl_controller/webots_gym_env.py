import gymnasium as gym
import numpy as np
import socket
import pickle
import struct
from gymnasium import spaces

HOST = "127.0.0.1"
PORT = 65432
MAX_SPEED = 6.4
ACTION_LIMIT = MAX_SPEED

def send_msg(sock, data):
    msg = pickle.dumps(data)
    sock.sendall(struct.pack('>I', len(msg)) + msg)

def recv_msg(sock):
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    return pickle.loads(recvall(sock, msglen))

def recvall(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

class WebotsGymnasiumEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_speed=6.4, render_mode=None):
        super().__init__()
        self.max_speed = max_speed
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((HOST, PORT))
        
        send_msg(self.conn, {"cmd": "reset"})
        resp = recv_msg(self.conn)
        first_obs = np.array(resp["obs"], dtype=np.float32)
        self.lidar_dim = len(first_obs) - 3
        self.last_obs = first_obs

        self.action_space = spaces.Box(
            low=np.array([-ACTION_LIMIT, -ACTION_LIMIT], dtype=np.float32),
            high=np.array([ACTION_LIMIT, ACTION_LIMIT], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.array([0.0]*self.lidar_dim + [-5.0, -5.0, -np.pi], dtype=np.float32),
            high=np.array([10.0]*self.lidar_dim + [5.0, 5.0, np.pi], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        send_msg(self.conn, {"cmd": "reset"})
        resp = recv_msg(self.conn)
        self.last_obs = np.array(resp["obs"], dtype=np.float32)
        return self.last_obs, {}

    def step(self, action):
        send_msg(self.conn, {"cmd": "step", "action": np.array(action, dtype=np.float32)})
        resp = recv_msg(self.conn)

        obs = np.array(resp["obs"], dtype=np.float32)
        reward = float(resp["reward"])
        terminated = bool(resp["done"])
        info = resp.get("info", {})

        self.last_obs = obs
        return obs, reward, terminated, False, info

    def close(self):
        try:
            send_msg(self.conn, {"cmd": "close"})
        except:
            pass
        self.conn.close()