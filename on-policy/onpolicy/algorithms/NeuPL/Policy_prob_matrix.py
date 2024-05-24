from onpolicy.algorithms.NeuPL.Nash_solver import npg
import numpy as np

np.set_printoptions(precision=3)
class Nash_matrix:
    def __init__(self, p1_space, p2_space):
        self.p1_space = p1_space
        self.p2_space = p2_space
        self.p1_num = len(p1_space)
        self.p2_num = len(p2_space)
        self.p1_prob_mat = np.zeros((self.p2_num,self.p1_num))
        self.p2_prob_mat = np.zeros((self.p1_num,self.p2_num))

    def update_prob_matrix_simple(self, p_mat):
        self.payoff_martix = np.array(p_mat)

        self.p1_prob_mat = np.zeros((self.p2_num, self.p1_num))
        self.p2_prob_mat = np.zeros((self.p1_num, self.p2_num))

        self.p1_prob_t = np.zeros(self.p1_num)
        self.p2_prob_t = np.zeros(self.p2_num)
        self.p1_payoff_line = np.zeros(self.p2_num)
        self.p2_payoff_line = np.zeros(self.p1_num)

        for i in range(1, min(self.p1_num,self.p2_num)):
            self.sub_matrix_simple(i, i)
        self.total_prob()
            
        print("p1_payoff_line = ", self.p1_payoff_line)
        print("p2_payoff_line = ", self.p2_payoff_line)

    def update_prob_matrix(self, p_mat, p1_map, p2_map):
        self.payoff_martix = np.array(p_mat)

        max_key1 = max(p1_map.keys())
        max_key2 = max(p2_map.keys())
        max_key = min(max_key1, max_key2)

        self.p1_prob_mat = np.zeros((max_key + 2, self.p1_num))
        self.p2_prob_mat = np.zeros((max_key + 2, self.p2_num))

        self.p1_prob_t = np.zeros(self.p1_num)
        self.p2_prob_t = np.zeros(self.p2_num)
        self.p1_payoff_line = np.zeros(max_key + 1)
        self.p2_payoff_line = np.zeros(max_key + 1)
        self.exploitability_avg = dict()
        self.exploitability_p1 = dict()
        self.exploitability_p2 = dict()

        for i in range(1, max_key + 2):
            p1_inx_list = []
            p2_inx_list = []
            if i > 1:
                p1_prob_prev = self.p1_prob_mat[i-1,:self.payoff_martix.shape[0]]
                p2_prob_prev = self.p2_prob_mat[i-1,:self.payoff_martix.shape[1]]

                p1_payoff = np.dot(self.payoff_martix[p1_map[i-1], :], p2_prob_prev)
                p2_payoff = np.dot(self.payoff_martix[:, p2_map[i-1]], p1_prob_prev)
                self.exploitability_avg[i-1] = (p1_payoff - p2_payoff)/2
                self.exploitability_p1[i-1] = p1_payoff - self.p1_payoff_line[i-2]
                self.exploitability_p2[i-1] = self.p1_payoff_line[i-2] - p2_payoff

            for key in range(i):
                if key in p1_map and p1_map[key] not in p1_inx_list:
                    p1_inx_list.append(p1_map[key])
            for key in range(i):
                if key in p2_map and p2_map[key] not in p2_inx_list:
                    p2_inx_list.append(p2_map[key])
            self.sub_matrix(i, p1_inx_list, p2_inx_list)

        # self.total_prob()
            
        print("p1_payoff_line = ", self.p1_payoff_line)
        print("p2_payoff_line = ", self.p2_payoff_line)

        print("average exploitability = ", self.exploitability_avg)

    def sub_matrix_simple(self, inx_p1, inx_p2):
        sub_mat = self.payoff_martix[0:inx_p1,:][ :, 0:inx_p2]
        payoff_mat_line = sub_mat.flatten()
        payoff_sub_total = np.zeros((len(payoff_mat_line), 2))
        for i in range(len(payoff_mat_line)):
            payoff_sub_total[i, 0] = payoff_mat_line[i]
            payoff_sub_total[i, 1] = -payoff_mat_line[i]
        sub_player_num = np.array([inx_p1, inx_p2])
        A_sub, payoff, *_ = npg(sub_player_num, payoff_sub_total)
        p1_max_inx = min(A_sub.shape[0], self.p1_num)
        p2_max_inx = min(A_sub.shape[0], self.p2_num)
        self.p1_prob_mat[inx_p2][0:p1_max_inx] = A_sub[:p1_max_inx,0]
        self.p2_prob_mat[inx_p1][0:p2_max_inx] = A_sub[:p2_max_inx,1]
        self.p1_payoff_line[inx_p2 -1] = payoff[0]
        self.p2_payoff_line[inx_p1 -1] = payoff[1]

    def caul_prob_from_payoff(self, payoff_mat):
        num_p1, num_p2 = payoff_mat.shape[0], payoff_mat.shape[1]
        payoff_mat_line = payoff_mat.flatten()
        payoff_sub_total = np.zeros((len(payoff_mat_line), 2))
        for i in range(len(payoff_mat_line)):
            payoff_sub_total[i, 0] = payoff_mat_line[i]
            payoff_sub_total[i, 1] = -payoff_mat_line[i]
        sub_player_num = np.array([num_p1, num_p2])
        A_sub, payoff, *_ = npg(sub_player_num, payoff_sub_total)
        p1_probs = A_sub[:num_p1,0]
        p2_probs = A_sub[:num_p2,1]
        payoff_p1 = payoff[0]
        payoff_p2 = payoff[1]
        return [p1_probs, p2_probs], [payoff_p1, payoff_p2]
        
    
    def sub_matrix(self, inx_i, p1_array, p2_array):
        sub_mat = self.payoff_martix[p1_array, :][:, p2_array]
        payoff_mat_line = sub_mat.flatten()
        payoff_sub_total = np.zeros((len(payoff_mat_line), 2))
        for i in range(len(payoff_mat_line)):
            payoff_sub_total[i, 0] = payoff_mat_line[i]
            payoff_sub_total[i, 1] = -payoff_mat_line[i]
        sub_player_num = np.array([len(p1_array), len(p2_array)])
        A_sub, payoff, *_ = npg(sub_player_num, payoff_sub_total)
        prob1_buffer = np.zeros(self.p1_num)
        prob2_buffer = np.zeros(self.p2_num)
        for i in range(len(p1_array)):
            prob1_buffer[p1_array[i]] = A_sub[i,0]
        for i in range(len(p2_array)):
            prob2_buffer[p2_array[i]] = A_sub[i,1]
        self.p1_prob_mat[inx_i] = prob1_buffer
        self.p2_prob_mat[inx_i] = prob2_buffer
        self.p1_payoff_line[inx_i -1] = payoff[0]
        self.p2_payoff_line[inx_i -1] = payoff[1]

    def total_prob(self):
        payoff_mat_line = self.payoff_martix.flatten()
        payoff_total = np.zeros((len(payoff_mat_line), 2))
        for i in range(len(payoff_mat_line)):
            payoff_total[i, 0] = payoff_mat_line[i]
            payoff_total[i, 1] = -payoff_mat_line[i]
        player_num = np.array([self.p1_num, self.p2_num])
        A_t, payoff, *_ = npg(player_num, payoff_total)
        self.p1_prob_t = A_t[:, 0]
        self.p2_prob_t = A_t[:, 1]
        self.p1_payoff_line[-1] = payoff[0]
        self.p2_payoff_line[-1] = payoff[1]


if __name__ == "__main__":
    p1_space = ["p" + str(i) for i in range(8)]
    p2_space = ["e" + str(i) for i in range(8)]
    # pay_mat =  np.array([
    #             [ 3.50321001e-01, -7.34169113e-01,  2.22944568e+00,
    #             4.71634326e-01,  2.57056157e-01,  3.50201174e-01,
    #             -2.44619366e-02, -2.19243492e+00],
    #             [-2.90057637e-02, -3.08137300e-02,  3.37563701e-01,
    #                 -1.21284720e+00, -9.44377806e-01,  1.25025123e+00,
    #                 -1.94884718e+00, -2.31928031e+00],
    #             [ 1.82452168e-01,  2.32347013e-01,  1.00006082e+00,
    #                 6.61900484e-02, -1.32178852e+00,  9.29789459e-01,
    #                 1.02049801e+00,  7.99337103e-02],
    #             [-1.56505601e+00,  4.26387557e-01, -1.66416447e+00,
    #                 6.52355889e-01,  9.24825933e-01,  2.39763257e-01,
    #                 8.61716302e-01, -9.48480984e-01],
    #             [-8.45394798e-02, -3.72808742e-01, -5.90034564e-01,
    #                 3.27059967e-01,  4.98490753e-05, -6.90361103e-01,
    #                 1.16208348e-03,  4.11490621e-01],
    #             [ 1.60394635e+00, -2.36454584e-01, -2.78064164e-01,
    #                 1.08263350e+00, -5.49189146e-02, -6.51553642e-01,
    #                 -7.08372132e-02,  6.76977806e-01],
    #             [ 9.83477746e-02,  2.02369089e+00,  4.22715691e-01,
    #                 1.00607711e+00,  9.11127266e-01,  1.19210187e+00,
    #                 -2.48628392e+00,  8.57732545e-01],
    #             [ 4.13736135e-02, -2.25835397e+00, -1.67020070e+00,
    #                 -6.50907737e-01,  5.94583697e-01, -1.61183039e+00,
    #                 5.81172323e-01, -6.91159125e-01]
    #         ])
    pay_mat = np.array([[1,2,3],[-1,3,2]])
    Nash_prob_test = Nash_matrix(p1_space, p2_space)
    Nash_prob_test.update_prob_matrix(pay_mat)
    print(Nash_prob_test.p1_prob_mat)
    print(Nash_prob_test.p2_prob_mat)

    print(Nash_prob_test.p1_prob_t)
    print(Nash_prob_test.p2_prob_t)

    print(Nash_prob_test.p1_payoff_line)
    print(Nash_prob_test.p2_payoff_line)