import numpy as np
import pandas as pd
import time

ct=0
N_STATES = 84   # 1维世界的宽度
offset = 0
ACTIONS = ['hold', 'change']     # 探索者的可用动作
EPSILON = 0.9   # 贪婪度 greedy
ALPHA = 0.1     # 学习率
GAMMA = 0.9    # 奖励递减值
MAX_EPISODES = 13   # 最大回合数



rr=[]
rr_profit=[]
rr_profit_post=[]
q_table_set=[]

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table 全 0 初始
        columns=actions,    # columns 对应的是行为名称
    )
    
    return table

# q_table:
"""
   hold   change
0   0.0    0.0
1   0.0    0.0
2   0.0    0.0
3   0.0    0.0
4   0.0    0.0
5   0.0    0.0
"""
# 在某个 state 地点, 选择行为
def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]  # 选出这个 state 的所有 action 值 -> nochange & buymore 列出來
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):  # 非貪婪 or 這個 state 還沒被探索過 (隨機選) (1-EPSILON)
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()    # 贪婪模式 EPSILON
    return action_name


def get_env_feedback(S, A, money,r_num,timestamp, profit):
    # This is how agent will interact with the environment
    # R: reward value
    
    if A == 'hold':
        S_ = S + 1
        # R = 當前value / 開始時value
        R = rr[r_num][timestamp+offset] 
        money *= R

    else: # A == 'change'
        S_ = S + 1
        R = rr[r_num][timestamp+offset] 
        profit = money*(R-1)
        if profit < 0: # 若虧損 則剩下的錢繼續投入
            money += profit
        else:
            money *= 1

    M = money
    P = profit

    if S == N_STATES - 2: # finish training
        S_ = 'terminal'
        return S_, 1, M, P

    return S_, R, M, P



def rl(i,flag):
    np.random.seed(5)

    if flag==False:
        q_table = build_q_table(N_STATES, ACTIONS)  # 初始 q table
    else:
        global q_table_set
        q_table=q_table_set[i]
    
    for episode in range(MAX_EPISODES):     # 回合 
        money = 1000000 # 初始金額 
        profit = 0
        step_counter = 0
        S = 0   # 回合初始位置
        is_terminated = False   # 是否回合结束

        while not is_terminated:
           
            A = choose_action(S, q_table)   # 选行为
            S_, R, M, P = get_env_feedback(S, A, money, i, episode, profit)  # 实施行为并得到环境的反馈
            q_predict = q_table.loc[S, A]    # 估算的(状态-行为)值
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   #  實際的(狀態-行為)值 (回合沒結束)
                
            else:
                q_target = R     #  實際的(狀態-行為)值 (回合結束)
                is_terminated = True    # terminate this episode
   
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  #  q_table 更新
            S = S_  # Agent移動到下一個 state
            money = M
            profit = P
           
            step_counter += 1
        #print("money: %d \n" %M)
    global rr_profit
    if(flag==False):
        rr_profit.append(int(money+profit-1000000))  
    else:
        rr_profit_post.append(int(money+profit-1000000)) 
    return q_table

import json
if __name__ == "__main__":
    file = json.load(open("./sp500_r_4mo.json","r"))
    lst=file['r']
    lst=[lst[i]+[1] for i in range(len(lst))]
    
    rr=lst
    tct=0
    for i in range(50):#先跑前50筆
        #tct+=1
        #print(tct)
        q_table_set.append(rl(i,False))#0個月
    #print('\r\nQ-table:\n')
    #print(q_table)     
    #print(rr_profit)
    for i in range(50):
        #tct+=1
        #print(tct)
        offset=42
        q_table_set[i]=rl(i,True)#2+2個月
    print(rr_profit_post)
    print(sum(rr_profit_post))
    for i in range(50):
        #tct+=1
        #print(tct)
        offset=42
        q_table=rl(i,False)#對照組
    print(rr_profit[50:])
    print(sum(rr_profit[50:]))
