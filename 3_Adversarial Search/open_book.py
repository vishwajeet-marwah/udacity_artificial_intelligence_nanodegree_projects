'''Create an open book of the best moves for every possible game state
   from an empty board to at least a depth of 4 plies'''


import time
import random, pickle


from isolation import Isolation
from isolation.isolation import _WIDTH, _HEIGHT,_ACTIONSET,Action

NUM_ROUNDS = 150

def build_table(num_rounds = NUM_ROUNDS):
    # Builds a table that maps from game state -> action
    # by choosing the action that accumulates the most
    # wins for the active player.   
    from collections import defaultdict, Counter
    book = defaultdict(Counter)
    print("Round :", end=' ')
    for index in range(num_rounds):
        state = Isolation()
        print(index + 1, end=' ')
        build_tree(state, book)
        #_print_data(book)
    print()
    return {k: max(v, key=v.get) for k, v in book.items()}

def build_tree(state, book, depth=5):
    if depth <= 0 or state.terminal_test():
        return -simulate(state)
    #action = random.choice(state.actions())
    #action = minimax_search(state, depth)
    '''if random.random() < 0.3:
        action = random.choice(state.actions())
    else:'''
    action = alpha_beta_search(state)
    reward = build_tree(state.result(action), book, depth - 1)
    
    '''sym_states_type = symmetric_states(state)
    sym_states = [sym_state[0] for sym_state in sym_states_type]
    for i,sym_state in enumerate(sym_states):
        if sym_state in book.keys():
            sym_type = sym_states_type[i][1]
            sym_action = symmetric_action(action, sym_type)
            book[sym_state][sym_action] += reward
            break  '''
    
    
    book[state][action] += reward
    return -reward


def simulate(state):
    while not state.terminal_test():
        state = state.result(random.choice(state.actions()))
    return -1 if state.utility(state.player()) < 0 else 1



def alpha_beta_search(state, depth=3):
    
    def min_value(state, alpha, beta, depth):
        if state.terminal_test(): return state.utility(state.player())
        if depth <= 0 : return custom_heuristic(state) # heuristic_score(state)
            
        value = float("inf")
        for action in state.actions():
            value = min(value, max_value(state.result(action), alpha, beta, depth-1))
            if value <= alpha: return value
            beta = min(beta, value)
        return value
        

    def max_value(state, alpha, beta, depth):
        
        if state.terminal_test(): return state.utility(state.player())
        if depth <= 0 : return custom_heuristic(state) # heuristic_score(state)

        value = float("-inf")
        for action in state.actions():
            value = max(value, min_value(state.result(action), alpha, beta, depth-1))
            if value >= beta: return value
            alpha = max(alpha, value)
        return value
    
    def heuristic_score(state):
        # A list containing the position of open liberties in the
        # neighborhood of the starting position
        own_loc = state.locs[state.player()]
        opp_loc = state.locs[1 - state.player()]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)
    
    
    
    def custom_heuristic(state):
        # A list containing the position of open liberties in the
        # neighborhood of the starting position
        own_loc = state.locs[state.player()]
        opp_loc = state.locs[1 - state.player()]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        
        
        # states away from edges from be encouraged, so the weight is bigger
        if distance(state) >= 2: 
            return 2*len(own_liberties) - len(opp_liberties)
        else: 
            return len(own_liberties) - len(opp_liberties)
        
        return len(own_liberties) - len(opp_liberties)

    
    
    def distance(state):
        """ minimum distance to the walls """
        own_loc = state.locs[state.ply_count % 2]
        x, y = own_loc // (_WIDTH + 2), own_loc % (_WIDTH + 2)

        return min(x, _WIDTH + 1 - x, y, _HEIGHT - 1 - y)
    
    
    
    alpha = float("-inf")
    beta = float("inf")
    best_score = float("-inf")
    best_move = None
    for action in state.actions():
        value = min_value(state.result(action), alpha, beta, depth-1)
        alpha = max(alpha, value)
        if value >= best_score:
            best_score = value
            best_move = action
        
    return best_move






def symmetric_positions(position, sym_type):
    row, col = position // (_WIDTH+2), position % (_WIDTH + 2)
    if sym_type == 'LR':
        return row * (_WIDTH+2) + (_WIDTH+1-col)
    elif sym_type == 'UD':
        return (_HEIGHT-1-row) * (_WIDTH+2) + col
    elif sym_type == 'LRUD':
        return (_HEIGHT-1-row) * (_WIDTH+2) + (_WIDTH+1-col)


def symmetric_states(state):
    """  return the states symmetric to the current one   """
    board = bin(state.board)[2:]
    board = [board[i*(_WIDTH+2):(i+1)*(_WIDTH+2)] for i in range(_HEIGHT)]

    # left - right:
    board_1 = eval('0b'+''.join([row[::-1] for row in board]))
    locs_1 = (symmetric_positions(state.locs[0],'LR') if state.locs[0] != None else None,symmetric_positions(state.locs[1],'LR') if state.locs[1] != None else None)
    state_1 = Isolation(board_1, ply_count=state.ply_count, locs=locs_1)
    # up - down:
    board_2 = eval('0b'+''.join(board[::-1]))
    locs_2 = (symmetric_positions(state.locs[0],'UD') if state.locs[0] != None else None,symmetric_positions(state.locs[1],'UD') if state.locs[1] != None else None)
    state_2 = Isolation(board_2, ply_count=state.ply_count, locs=locs_2)

    # left - right and up - down:
    board_3 = eval('0b'+''.join([row[::-1] for row in board[::-1]]))
    locs_3 = (symmetric_positions(state.locs[0],'LRUD') if state.locs[0] != None else None,symmetric_positions(state.locs[1],'LRUD') if state.locs[1] != None else None)
    state_3 = Isolation(board_3, ply_count=state.ply_count, locs=locs_3)

    return [(state_1,'LR'),(state_2,'UD'),(state_3,'LRUD')]


sym_dict_LR = {Action.NNE:Action.NNW,Action.ENE:Action.WNW,Action.ESE:Action.WSW,Action.SSE:Action.SSW,Action.SSW:Action.SSE,Action.WSW:Action.ESE,Action.WNW:Action.ENE,Action.NNW:Action.NNE}
sym_dict_UD = {Action.NNE:Action.SSE,Action.ENE:Action.ESE,Action.ESE:Action.ENE,Action.SSE:Action.NNE,Action.SSW:Action.NNW,Action.WSW:Action.WNW,Action.WNW:Action.WSW,Action.NNW:Action.SSW}
sym_dict_LRUD = {Action.NNE:Action.SSW,Action.ENE:Action.WSW,Action.ESE:Action.WNW,Action.SSE:Action.NNW,Action.SSW:Action.NNE,Action.WSW:Action.ENE,Action.WNW:Action.ESE,Action.NNW:Action.SSE}

def symmetric_action(action, sym_type):
    """ return the symmetric action according to the symmetry type sym_type """

    if action not in _ACTIONSET:
        return symmetric_positions(action,sym_type)
    elif sym_type == 'LR':
        return sym_dict_LR[action]
    elif sym_type == 'UD':
        return sym_dict_UD[action]
    elif sym_type == 'LRUD':
        return sym_dict_LRUD[action]
    else:
        raise ValueError(" The value of sym_type is illegal")








def _print_data(data):
    for state, action in data.items():
        print("Player Id: {}".format(state.player()))
        print("Book State: {}".format(state))
        print("Action: {}".format(dict(action)))
    print()
    return None




if __name__ == "__main__":
    print("Enter open_book __main__")
    start_time = time.time()
    open_book_data = build_table(NUM_ROUNDS)
    end_time = time.time()
    #print("Total time for {} rounds: {} seconds.".format(NUM_ROUNDS,end_time - start_time))
    # data for open book - to be written in data.pickle file   
    print("Open Book Data for data.pickle file")
    for key, value in open_book_data.items():
        print("Book State: {}".format(key))
        print("Action: {}".format(value))  
    
    with open("data.pickle", 'wb') as f:
        pickle.dump(open_book_data, f)