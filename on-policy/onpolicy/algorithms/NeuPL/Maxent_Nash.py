import numpy as np
from tqdm import tqdm

class Maxent_NashSolver:
    # only for two-player & zero-sum game
    def __init__(self, p1_space, p2_space):
        self.p1_space = p1_space
        self.p2_space = p2_space

        index_p1 = 0
        self.p1_to_index = {}
        self.prob1 = {}
        for action in self.p1_space:
            self.p1_to_index[action] = index_p1
            self.prob1[action] = 0
            index_p1 += 1
            
        index_p2 = 0
        self.p2_to_index = {}
        self.prob2 = {}
        for action in self.p2_space:
            self.p2_to_index[action] = index_p2
            self.prob2[action] = 0
            index_p2 += 1

        self.prob_all = {}
        for i in range(len(p1_space)):
            for j in range(len(p2_space)):
                joint_action = (p1_space[i], p2_space[j])
                self.prob_all[joint_action] = 0

        self.dual_variables = {}

        #Joint action space
        self.p1_joint = []
        for i in range(len(p1_space)):
            for j in range(len(p1_space)):
                if j != i:
                    joint_action = (1, p1_space[i], p1_space[j])
                    self.p1_joint.append(joint_action)
                    self.dual_variables[joint_action] = 0

        self.p2_joint = []
        for i in range(len(p2_space)):
            for j in range(len(p2_space)):
                if j != i:
                    joint_action = (2, p2_space[i], p2_space[j])
                    self.p2_joint.append(joint_action)
                    self.dual_variables[joint_action] = 0


    def set_payoff_martix(self, payoff_matrix):
        self.payoff_matrix = payoff_matrix

    def payoff(self, action_1, action_2):
        return self.payoff_matrix[self.p1_to_index[action_1]][self.p2_to_index[action_2]]
    
    def payoff_gain(self, alt_action, action, action_op, player=1):
        #Calculate M(alt_action, action') & M(action, action')
        diff = 0 
        #opponents action space
            
        if player == 1:
            M_alt = self.payoff(alt_action, action_op)
            M_act = self.payoff(action, action_op)
            diff += M_alt - M_act
        else:
            M_alt = -self.payoff(action_op, alt_action)
            M_act = -self.payoff(action_op, action)
            diff += M_alt - M_act
        return diff
    
    #Return Z(lambda) of dual variables
    def Z(self):
        sum_Z = 0
        
        #Split the sums
        for i_t in range(len(self.p1_space)):
            action_p1 = self.p1_space[i_t]
            for j_k in range(len(self.p2_space)):
                action_p2 = self.p2_space[j_k]
                loop_sum = 0
                for k in range(len(self.p1_space)):
                    if self.p1_space[k] != action_p1:
                        loop_sum += self.dual_variables[(1, action_p1, self.p1_space[k])] * self.payoff_gain(self.p1_space[k], action_p1, action_p2, player=1)
                for k in range(len(self.p2_space)):
                    if self.p2_space[k] != action_p2:
                        loop_sum += self.dual_variables[(2, action_p2, self.p2_space[k])] * self.payoff_gain(self.p2_space[k], action_p2, action_p1, player=2)   
                sum_Z += np.exp(-loop_sum)                 
            
        return sum_Z
    
    #Get mixed strategy from dual variables
    def P(self, action_p1, action_p2):
        sum_one = 0
        for k in range(len(self.p1_space)):
            if self.p1_space[k] != action_p1:
                sum_one += self.dual_variables[(1, action_p1, self.p1_space[k])] * self.payoff_gain(self.p1_space[k], action_p1, action_p2, player=1)
        for k in range(len(self.p2_space)):
            if self.p2_space[k] != action_p2:
                sum_one += self.dual_variables[(2, action_p2, self.p2_space[k])] * self.payoff_gain(self.p2_space[k], action_p2, action_p1, player=2)   

        log_P = -sum_one - np.log(self.Z_lambda)
        return np.exp(log_P)
    
    #Regret Calculation
    def regret_both(self, action, action_prime, player=1):
        p = 0
        n = 0
        if player == 1:
            for i in range(len(self.p2_space)):
                P_p1 = self.P(action, self.p2_space[i])
                p_gain = self.payoff_gain(action_prime, action, self.p2_space[i], player)
                p += P_p1 * max(0, p_gain)
                n += P_p1 * max(0, -p_gain)
        else:
            for i in range(len(self.p1_space)):
                P_p1 = self.P(self.p1_space[i], action)
                p_gain = self.payoff_gain(action_prime, action, self.p1_space[i], player)
                p += P_p1 * max(0, p_gain)
                n += P_p1 * max(0, -p_gain)
            
        return p + 1e-8, n + 1e-8
    
    def abs_gain(self, action, action_prime, action_op, player=1):
        total = 0
        if player == 1:
                p_gain = abs(self.payoff(action_prime, action_op) - self.payoff(action, action_op))
                total = p_gain
        else:
                p_gain = abs((-self.payoff(action_op, action_prime)) - (-self.payoff(action_op, action)))
                total = p_gain
        
        return total

    def lower_bound_c(self):
        bound = 0
        max_gain =-1000000
        for action_p1 in self.p1_space:
            for action_p2 in self.p2_space:
                bound = 0
                for action_prime in self.p1_space:
                    a_gain = self.abs_gain(action_p1, action_prime, action_p2, player=1)
                    bound += a_gain
                
                for action_prime in self.p2_space:
                    a_gain = self.abs_gain(action_p2, action_prime, action_p1, player=2)
                    bound += a_gain
                if bound > max_gain:
                    max_gain = bound
        return max_gain
    
    def sort_dictionary(self, d):
        sorted_x = sorted(d.items(), key=lambda kv: kv[1])
        return sorted_x
    
    #Not log_grad.. dynamic gradient ascent avoids divide by 0
    def log_grad_descent(self, rounds=10, gamma=0.0):
        c= self.lower_bound_c()
        
        for it in tqdm(range(rounds)):
            self.Z_lambda = self.Z()
            self.step_dict = {}
            for pair in self.p1_joint:
                _, action, action_prime = pair
                r_pos, r_neg = self.regret_both(action, action_prime, player=1)
                
                reg = gamma*self.dual_variables[pair]
                term = ((r_pos - reg)/(r_pos + r_neg)) - (1/2)

                step = (1/c)*term

                self.step_dict[pair] = step
            
            for pair in self.p2_joint:
                _, action, action_prime = pair
                r_pos, r_neg = self.regret_both(action, action_prime, player=2)

                reg = gamma*self.dual_variables[pair]
                term = ((r_pos - reg)/(r_pos + r_neg)) - (1/2)
                
                step = (1/c)*term

                self.step_dict[pair] = step
                    
            for pair in self.p1_joint:
                self.dual_variables[pair] = max(0, self.dual_variables[pair] + self.step_dict[pair])
            
            for pair in self.p2_joint:
                self.dual_variables[pair] = max(0, self.dual_variables[pair] + self.step_dict[pair])

        self.set_zero_probs()
        self.Z_lambda = self.Z()

        for action_p1 in self.p1_space:
            for action_p2 in self.p2_space:
                self.prob_all[(action_p1,action_p2)] = self.P(action_p1, action_p2)
                self.prob1[action_p1] += self.prob_all[(action_p1,action_p2)]
                self.prob2[action_p2] += self.prob_all[(action_p1,action_p2)]

        prob1_array = np.array(list(self.prob1.values()), dtype=float).reshape(-1, 1)
        prob2_array = np.array(list(self.prob2.values()), dtype=float).reshape(-1, 1)
        self.nash_payoff = prob1_array.T @ self.payoff_matrix[0:len(self.p1_space), 0:len(self.p2_space)] @ prob2_array

    def set_zero_probs(self):
        for key in self.prob1:
            self.prob1[key] = 0
        for key in self.prob2:
            self.prob2[key] = 0

if __name__ == "__main__":
    p1_space = ["p"+str(i) for i in range(7)]
    p2_space = ["e"+str(i) for i in range(5)]
    payoff_m = np.array([
                        [ 0.67149713, -0.30344092, -2.94428416, -0.24144704,  1.09326567],
                        [-1.20748692,  0.29387147,  1.43838029,  0.31920674,  1.1092733 ],
                        [ 0.71723865, -0.7872828 ,  0.32519054,  0.3128586 , -0.86365282],
                        [ 1.63023529,  0.88839563, -0.75492832, -0.86487992,  0.07735909],
                        [ 0.48889377, -1.14707011,  1.37029854, -0.0300513 , -1.21411704],
                        [ 1.03469301, -1.06887046, -1.71151642, -0.16487902, -1.11350074],
                        [ 0.72688513, -0.80949869, -0.10224245,  0.62770729, -0.00684933]
                        ])
    #payoff_m = np.log(payoff_m/(1-payoff_m))
    Nash_test = Maxent_NashSolver(p1_space, p2_space)
    Nash_test.set_payoff_martix(payoff_m)
    Nash_test.log_grad_descent(rounds=20000)
    print(Nash_test.prob1)
    print(Nash_test.prob2)
    print(Nash_test.nash_payoff)
    #print(Nash_test.step_dict)