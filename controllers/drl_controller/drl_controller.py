from controller import Lidar, Motor, GPS, InertialUnit, Supervisor
import random, socket, pickle, struct, os
import numpy as np

MAP_SIZE = 10.0
RESOLUTION = 0.025
TIME_STEP = 32
MAX_SPEED = 6.4
MAX_STEPS = 3000
COLLISION_THRESHOLD = 0.1 
DANGER_THRESHOLD = 0.18
GOAL_REWARD = 50
IN_PLACE_PENALTY = -1
AVOIDANCE_REWARD = +1
HOST = "127.0.0.1"
PORT = 65432

def send_msg(sock, data): 
    msg = pickle.dumps(data)
    sock.sendall(struct.pack('>I', len(msg)) + msg)

def recv_msg(sock):
    raw_msglen = recvall(sock, 4)
    if not raw_msglen: return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    return pickle.loads(recvall(sock, msglen))

def recvall(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet: return None
        data += packet
    return data

def get_obs(lidar, gps, imu):
    scan = np.array(lidar.getRangeImage(), dtype=np.float32)
    scan = np.nan_to_num(scan, nan=10.0, posinf=10.0, neginf=0.0)
    scan = scan[::10]  
    pos = gps.getValues()
    yaw = imu.getRollPitchYaw()[2]
    return np.concatenate([scan, [pos[0], pos[1], yaw]])

def get_last_episode(csv_path="training_summary_log.csv"):
    if not os.path.exists(csv_path): return 0
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        return df["Episode"].max()
    except: return 0



def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    lidar = robot.getDevice("lidar"); lidar.enable(timestep)
    gps = robot.getDevice("gps"); gps.enable(timestep)
    imu = robot.getDevice("inertial unit"); imu.enable(timestep)
    left_motor = robot.getDevice("wheel_left_joint")
    right_motor = robot.getDevice("wheel_right_joint")
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))

    print(" Waiting for RL agent...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT)); s.listen(1)
    conn, addr = s.accept()
    print(" Agent connected from", addr)

    USE_RANDOM_GOAL = True
    goal = np.array([1.0, 1.0])
    total_episodes = get_last_episode()
    total_goals = total_crashes = total_timeouts = total_steps_across_episodes = 0
    outcome = None
    episode_angular_velocities = []
    prev_position = np.zeros(2)  
    stuck_counter = 0

    def min_lidar_on_left(scan, threshold=COLLISION_THRESHOLD):
        left_half = scan[:len(scan)//2]
        return np.min(left_half) < threshold

    def min_lidar_on_right(scan, threshold=COLLISION_THRESHOLD):
        right_half = scan[len(scan)//2:]
        return np.min(right_half) < threshold

    while robot.step(TIME_STEP) != -1:
        data = recv_msg(conn)
        if data is None: break

        cmd = data.get("cmd")
        if cmd == "reset":
            total_episodes += 1
            step_count = 0
            outcome = None
            episode_angular_velocities = []
            


            prev_action = np.array([0.0, 0.0]) 

            if USE_RANDOM_GOAL:
                goal = np.random.uniform(-3.5, 3.5, size=2)
                print(f" New Goal: {goal}")

            translation = robot.getSelf().getField("translation")
            rotation = robot.getSelf().getField("rotation")
            spawn = np.random.uniform(-3.0, 3.0, size=2)
            translation.setSFVec3f([spawn[0], spawn[1], 0.095])
            rotation.setSFRotation([0, 0, 1, 1.5708])
            for _ in range(5): robot.step(TIME_STEP)

            obs = get_obs(lidar, gps, imu)
            prev_dist = np.linalg.norm(obs[-3:-1] - goal)
            robot.prev_dist_to_goal = prev_dist
            x, y = obs[-3], obs[-2]
            prev_position[:] = [x, y] 
            stuck_counter = 0
            send_msg(conn, {"obs": obs})
            continue

        elif cmd == "step":
            action = np.clip(data["action"], -MAX_SPEED, MAX_SPEED)
            angular_velocity = abs(action[0] - action[1])
            episode_angular_velocities.append(angular_velocity)
            left_motor.setVelocity(float(action[0]))
            right_motor.setVelocity(float(action[1]))
            robot.step(TIME_STEP)

            obs = get_obs(lidar, gps, imu)
            step_count += 1

            x, y = obs[-3], obs[-2]
            
            dist = np.linalg.norm([x - goal[0], y - goal[1]])
            min_lidar = np.min(obs[:-3])    
        
            cur_position = np.array([x, y])
            if np.linalg.norm(cur_position - prev_position) < 0.05:
                stuck_counter += 1
            else:
                stuck_counter = 0
            prev_position = cur_position
            
            reward = 0.0
            
            # Progress reward
            progress_reward = (robot.prev_dist_to_goal - dist) * 8.0
            robot.prev_dist_to_goal = dist
            reward += progress_reward
            
            # Stuck penalty
            if stuck_counter > 50:
                reward -= 2.0
            
            # Goal alignment
            goal_vec = np.array([goal[0] - x, goal[1] - y])
            goal_angle = np.arctan2(goal_vec[1], goal_vec[0])
            yaw = obs[-1]
            angle_diff = np.arctan2(np.sin(goal_angle - yaw), np.cos(goal_angle - yaw))
            reward += 2.0 * np.cos(angle_diff)
            
            # Forward motion bonus
            if action[0] > 0 and action[1] > 0:
                reward += 0.1
            
            # Mild turning 
            if 0.2 < angular_velocity < 2.0:
                reward += 0.05
            
            # Penalize erratic action 
            delta_action = np.abs(np.array(action) - prev_action)
            reward -= 0.03 * np.sum(delta_action)
            prev_action = np.array(action)
            
            # Penalize spinning
            if angular_velocity > 3.0:
                reward -= 0.1 * angular_velocity
            
            # Danger zone penalty
            if DANGER_THRESHOLD > min_lidar >= COLLISION_THRESHOLD:
                reward -= 0.5
            
            # Proximity bonus
            if dist < 1.5:
                reward += (1.5 - dist) * 2.0
            
            # Active movement bonus
            if np.linalg.norm(action) > 0.1:
                reward += 0.05
            
            # Time penalty
            reward -= 0.002
            
            done = False
            if min_lidar < COLLISION_THRESHOLD:
                reward -= 10
                done = True
                outcome = "crash"
                total_crashes += 1
                print(f"⚠️ Crash @ step {step_count}")
                
            
            elif dist < 0.5:
                reward += GOAL_REWARD
                if step_count < 1500:
                    reward += 10.0 
                done = True
                outcome = "goal"
                total_goals += 1
                print(f" Goal reached in {step_count} steps (dist: {dist:.2f})")
            
            elif step_count >= MAX_STEPS:
                reward -= 3
                done = True
                outcome = "timeout"
                total_timeouts += 1
                print("Timeout")
            
            reward = np.clip(reward, -10.0, 10.0)


            if done:
                total_steps_across_episodes += step_count
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)
                mean_ang_vel = np.mean(episode_angular_velocities) if episode_angular_velocities else 0.0
                send_msg(conn, {
                    "obs": obs,
                    "reward": reward,
                    "done": True,
                    "info": {
                        "done_reason": outcome,
                        "angular_var": mean_ang_vel
                    }
                })
            else:
                send_msg(conn, {"obs": obs, "reward": reward, "done": False, "info": {}})

            print(f"[Step {step_count}] Action: {action}, Reward: {reward:.2f}, Dist: {dist:.2f}, LidarMin: {min_lidar:.2f}, Done: {done}")

        elif cmd == "close":
            break

    # Final Summary
    print("\n===== TRAINING SUMMARY =====")
    print(f"Total Episodes : {total_episodes}")
    print(f"Goals Reached  : {total_goals}")
    print(f"Crashes        : {total_crashes}")
    print(f"Timeouts       : {total_timeouts}")
    if total_episodes > 0:
        avg = total_steps_across_episodes / total_episodes
        print(f"Avg Steps/Ep   : {avg:.2f}")
    print("=================================")

    left_motor.setVelocity(0)
    right_motor.setVelocity(0)
    conn.close(); s.close()

if __name__ == "__main__":
    main()