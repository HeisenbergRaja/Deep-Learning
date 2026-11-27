# Program 23: Reinforcement learning: Traffic signal optimization agent using Q-learning
# Question: A smart city wants to reduce traffic congestion by developing an intelligent traffic signal system. Design and implement a reinforcement learning agent that learns optimal traffic signal actions based on environmental states and reward feedback. (BL6 – Create)

import numpy as np

np.random.seed(42)

# Environment: Asymmetric intersection (North-South rush hour vs East-West light traffic)
# States: Discretized queue congestion: (NS_Level, EW_Level) where 0=Low, 1=Medium, 2=High -> 9 total states
# Actions: 0 = Green NS (Red EW), 1 = Green EW (Red NS)
# Reward: Negative total waiting cars -(q_NS + q_EW)
n_states = 9
n_actions = 2
Q_table = np.zeros((n_states, n_actions))

alpha_rl = 0.1     # Learning rate
gamma_rl = 0.9     # Discount factor
epsilon_rl = 1.0   # Exploration rate
episodes = 500
steps_per_ep = 30

def state_encoder(q_ns, q_ew):
    lvl_ns = 0 if q_ns <= 5 else (1 if q_ns <= 12 else 2)
    lvl_ew = 0 if q_ew <= 5 else (1 if q_ew <= 12 else 2)
    return lvl_ns * 3 + lvl_ew

# Q-Learning Training Loop
for ep in range(episodes):
    q_ns = np.random.randint(2, 10)
    q_ew = np.random.randint(1, 6)
    
    for t in range(steps_per_ep):
        state = state_encoder(q_ns, q_ew)
        
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon_rl:
            action = np.random.choice([0, 1])
        else:
            action = int(np.argmax(Q_table[state]))
            
        # Environment dynamics: Green road clears up to 4 vehicles
        if action == 0:   # Green NS
            q_ns = max(0, q_ns - 4)
        else:             # Green EW
            q_ew = max(0, q_ew - 4)
            
        # Rush-hour vehicle arrivals (asymmetric traffic distribution)
        q_ns += np.random.randint(2, 6)  # Heavier NS flow
        q_ew += np.random.randint(0, 3)  # Lighter EW flow
        
        reward = -(q_ns + q_ew)
        next_state = state_encoder(q_ns, q_ew)
        
        # Bellman equation Q-value update
        best_future_q = np.max(Q_table[next_state])
        Q_table[state, action] += alpha_rl * (reward + gamma_rl * best_future_q - Q_table[state, action])
        
    epsilon_rl = max(0.02, epsilon_rl * 0.99)

print("\n--- Program 23: Reinforcement Learning for Traffic Control ---")
state_desc = [
    "NS:Low, EW:Low ", "NS:Low, EW:Med ", "NS:Low, EW:High",
    "NS:Med, EW:Low ", "NS:Med, EW:Med ", "NS:Med, EW:High",
    "NS:High,EW:Low ", "NS:High,EW:Med ", "NS:High,EW:High"
]
print("Learned Policy Table (Q-Values):")
print(f"{'State':<18}{'Q(Green NS)':<15}{'Q(Green EW)':<15}{'Optimal Action'}")
print("-" * 62)
for s in range(n_states):
    opt_act = "Green NS" if np.argmax(Q_table[s]) == 0 else "Green EW"
    print(f"{state_desc[s]:<18}{Q_table[s, 0]:<15.2f}{Q_table[s, 1]:<15.2f}{opt_act}")

# Policy Benchmark Simulation: Compare RL Agent vs Fixed-Time Signal
eval_runs = 50
fixed_total_cars, rl_total_cars = 0, 0

for _ in range(eval_runs):
    q_ns_f, q_ew_f = 8, 4
    q_ns_r, q_ew_r = 8, 4
    for step in range(30):
        # 1. Fixed-time signal (blindly alternates green light every step)
        fixed_act = step % 2
        q_ns_f = max(0, q_ns_f - 4) if fixed_act == 0 else q_ns_f
        q_ew_f = max(0, q_ew_f - 4) if fixed_act == 1 else q_ew_f
        
        # 2. RL Intelligent Agent
        rl_act = int(np.argmax(Q_table[state_encoder(q_ns_r, q_ew_r)]))
        q_ns_r = max(0, q_ns_r - 4) if rl_act == 0 else q_ns_r
        q_ew_r = max(0, q_ew_r - 4) if rl_act == 1 else q_ew_r
        
        arr_ns, arr_ew = np.random.randint(2, 6), np.random.randint(0, 3)
        q_ns_f += arr_ns; q_ew_f += arr_ew
        q_ns_r += arr_ns; q_ew_r += arr_ew
        fixed_total_cars += (q_ns_f + q_ew_f)
        rl_total_cars += (q_ns_r + q_ew_r)

avg_fixed_queue = fixed_total_cars / (eval_runs * 30)
avg_rl_queue = rl_total_cars / (eval_runs * 30)
reduction = ((avg_fixed_queue - avg_rl_queue) / avg_fixed_queue) * 100

print(f"\n[Performance Benchmark over {eval_runs} Episodes]:")
print(f"- Fixed-Time Signal  : Average Intersection Queue = {avg_fixed_queue:.2f} cars")
print(f"- RL Q-Learning Agent: Average Intersection Queue = {avg_rl_queue:.2f} cars")
print(f"- Congestion Reduction: {reduction:.1f}% reduction in waiting vehicles!")

# Output:
# Learned Policy Table (Q-Values):
# State             Q(Green NS)    Q(Green EW)    Optimal Action
# --------------------------------------------------------------
# NS:Low, EW:Low    -159.17        -161.82        Green NS
# NS:Low, EW:Med    -187.16        -162.05        Green EW
# NS:Low, EW:High   -221.36        -222.50        Green NS
# NS:Med, EW:Low    -184.20        -191.96        Green NS
# NS:Med, EW:Med    -216.34        -195.92        Green EW
# NS:Med, EW:High   -240.34        -236.29        Green EW
# NS:High,EW:Low    -221.47        -261.00        Green NS
# NS:High,EW:Med    -262.26        -232.31        Green EW
# NS:High,EW:High   -261.50        -260.27        Green EW
# [Performance Benchmark over 50 Episodes]:
# - Fixed-Time Signal  : Average Intersection Queue = 31.52 cars
# - RL Q-Learning Agent: Average Intersection Queue = 19.92 cars
# - Congestion Reduction: 36.8% reduction in waiting vehicles!

# ======================================================================
