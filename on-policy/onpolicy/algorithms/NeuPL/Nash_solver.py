import numpy as np
from scipy.optimize import minimize

def gamer(n, Us, p, I, s, ub, lb, x0, Aeq, beq, pay, U):

    def myfun(x):
        Funct = np.sum(x[s:])
        for i in range(p):
            prod = np.prod(x[I[i, :]])  
            Funct -= Us[i] * prod
        #print(Funct)
        return Funct

    def confun(x):
        C = np.zeros(s)
        for i in range(s):
            C[i] = -x[pay[i]] 
            for t in range(n):
                add = 0
                for j in range(p):
                    prd = 1
                    for k in range(n):
                        if i == I[j, k]:  
                            prd *= U[j, k]
                        else:
                            prd *= x[I[j, k]]  
                    if I[j, t] != i:
                        prd = 0
                    add += prd
                C[i] += add
        #print(-C)
        return -C

    constraints = [{'type': 'ineq', 'fun': confun}, {'type': 'eq', 'fun': lambda x: np.matmul(Aeq, x) - beq}]
    options = {'disp': False}

    res = minimize(myfun, x0, method='SLSQP', bounds=list(zip(lb, ub)),
                   constraints=constraints, options=options)

    return res.x, res.fun, res.success, res

def npg(M, U):
    p = np.prod(M)
    n = len(M)
    s = np.sum(M)
    A = np.zeros((max(M), n))
    payoff = np.zeros(n)

    if p != U.shape[0] or n != U.shape[1]:
        raise ValueError('Dimension mismatch!')

    P = np.zeros(n)
    N = np.zeros(n)
    P[-1] = 1
    for i in range(n-2, -1, -1):
        P[i] = P[i+1] * M[i+1]
    P = np.round(P).astype(int)

    N = p // P
    N = np.round(N).astype(int)
    #print(N, P)

    x0 = np.zeros(s)
    k = 0
    for i in range(n):
        for j in range(M[i]):
            x0[k] = 1 / M[i]
            k += 1

    Us = np.sum(U, axis=1)
    V = np.prod([(1 / mi) ** mi for mi in M])
    x0 = np.concatenate((x0, V * np.sum(U, axis=0)))

    Aeq = np.zeros((n, s+n))
    cnt = 0
    for i in range(n):
        if i != 0:
            cnt += M[i-1]
        for j in range(s):
            if cnt-1 < j <= np.sum(M[:i+1])-1:
                Aeq[i, j] = 1

    beq = np.ones(n)
    I = np.ones((p, n), dtype=int)
    counter = 0
    count = 0
    for i in range(n):
        for j in range(N[i]):
            for k in range(P[i]):
                I[count%p][ count // p] = counter
                count += 1
            counter += 1
            if i != 0 and counter >= np.sum(M[:i+1]):
                counter -= M[i]

    lb = np.zeros(s+n)
    ub = np.ones(s+n)
    pay = np.zeros(s, dtype=int)
    counter = 0
    for i in range(n):
        for j in range(M[i]):
            pay[counter] = i + s
            counter += 1

    lb[s:] = -np.inf
    ub[s:] = np.inf
    #print(I)
    x, fval, exitflag, output = gamer(n, Us, p, I, s, ub, lb, x0, Aeq, beq, pay, U)

    count = 0
    for i in range(n):
        for j in range(M[i]):
            A[j, i] = abs(x[count])
            count += 1
        payoff[i] = x[s+i]

    iterations = output['nit']
    err = abs(fval)

    return A, payoff, iterations, err

if __name__ == "__main__":
    # Define the number of strategies for each player
    M = [15, 11]

    # Initialize a payoff matrix with random values for each player
    #U_single = np.random.randn(*M)
    U_single=np.array([
                    [-1.30768829630527, 0.671497133608081, 1.43838029281510, -0.863652821988714, 0.0859311331754255, 1.41931015064255, 0.187331024578940, -2.13835526943994, -0.272469409250188, 0.0228897927516298, -0.133217479507735],
                    [-0.433592022305684, -1.20748692268504, 0.325190539456198, 0.0773590911304249, -1.49159031063761, 0.291584373984183, -0.0824944253709554, -0.839588747336614, 1.09842461788862, -0.261995434966092, -0.714530163787158],
                    [0.342624466538650, 0.717238651328839, -0.754928319169703, -1.21411704361541, -0.742301837259857, 0.197811053464361, -1.93302291785099, 1.35459432800464, -0.277871932787639, -1.75021236844679, 1.35138576842666],
                    [3.57839693972576, 1.63023528916473, 1.37029854009523, -1.11350074148676, -1.06158173331999, 1.58769908997406, -0.438966153934773, -1.07215528838425, 0.701541458163284, -0.285650971595330, -0.224771056052584],
                    [2.76943702988488, 0.488893770311789, -1.71151641885370, -0.00684932810334806, 2.35045722400204, -0.804465956349547, -1.79467884145512, 0.960953869740567, -2.05181629991115, -0.831366511567624, -0.589029030720801],
                    [-1.34988694015652, 1.03469300991786, -0.102242446085491, 1.53263030828475, -0.615601881466894, 0.696624415849607, 0.840375529753905, 0.124049800003193, -0.353849997774433, -0.979206305167302, -0.293753597735416],
                    [3.03492346633185, 0.726885133383238, -0.241447041607358, -0.769665913753682, 0.748076783703985, 0.835088165072682, -0.888032082329010, 1.43669662271894, -0.823586525156853, -1.15640165566400, -0.847926243637934],
                    [0.725404224946106, -0.303440924786016, 0.319206739165502, 0.371378812760058, -0.192418510588264, -0.243715140377952, 0.100092833139322, -1.96089999936503, -1.57705702279920, -0.533557109315987, -1.12012830124373],
                    [-0.0630548731896562, 0.293871467096658, 0.312858596637428, -0.225584402271252, 0.888610425420721, 0.215670086403744, -0.544528929990548, -0.197698225974150, 0.507974650905946, -2.00263573588306, 2.52599969211831],
                    [0.714742903826096, -0.787282803758638, -0.864879917324457, 1.11735613881447,	-0.764849236567874,	-1.16584393148205,	0.303520794649354,	-1.20784548525980,	0.281984063670556,	0.964229422631628,	1.65549759288735],
                    [-0.204966058299775,	0.888395631757642,	-0.0300512961962686,	-1.08906429505224,	-1.40226896933876,	-1.14795277889859,	-0.600326562133734,	2.90800803072936,	0.0334798822444514,	0.520060101455458,	0.307535159238252],
                    [-0.124144348216312,	-1.14707010696915,	-0.164879019209038,	0.0325574641649735,	-1.42237592509150,	0.104874716016494,	0.489965321173948,	0.825218894228491,	-1.33367794342811,	-0.0200278516425381,	-1.25711835935205],
                    [1.48969760778547,	-1.06887045816803,	0.627707287528727,	0.552527021112224,	0.488193909859941,	0.722254032225002,	0.739363123604474,	1.37897197791661,	1.12749227834159,	-0.0347710860284830,-0.865468030554804],
                    [1.40903448980048,	-0.809498694424876,	1.09326566903948,	1.10061021788087,	-0.177375156618825,	2.58549125261624,	1.71188778298155,	-1.05818025798736,	0.350179410603312,	-0.798163584564142,	-0.176534114231451],
                    [1.41719241342961,	-2.94428416199490,	1.10927329761440,	1.54421189550395,	-0.196053487807333,	-0.666890670701386,	-0.194123535758265,	-0.468615581100624,	-0.299066030332982,	1.01868528212858,	0.791416061628634]
    ])

    # Transpose and reshape the payoff matrix for the two-player game
    U_t = U_single
    U_line = U_single.flatten()
    #print(U_line)

    # Create a total payoff matrix with positive payoffs for player 1 and negative for player 2
    U_total = np.zeros((len(U_line), 2))
    for i in range(len(U_line)):
        U_total[i, 0] = U_line[i]
        U_total[i, 1] = -U_line[i]

    # Call the npg function to compute the Nash Equilibrium
    # This assumes that the npg function has been previously translated and is available in the scope
    A, payoff, iterations, err = npg(M, U_total)

    print(A)
    print(payoff)
    print(iterations)
    print(err)