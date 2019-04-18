
from sample_players import DataPlayer

from isolation.isolation import _WIDTH, _HEIGHT

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    
    
    def __init__(self, player_id):
        super().__init__(player_id)
    
    
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        
        import random
        
        if state.ply_count < 4:
            if self.data is not None and state in self.data:
                self.queue.put(self.data[state])
            else:
                self.queue.put(random.choice(state.actions()))
        else:
            ''' alpha - beta pruning with iterative deepening  '''
            depth_limit = 5
            for depth in range(1, depth_limit):
                best_move = self.alpha_beta_search(state, depth)
            self.queue.put(best_move)
            
        
    def alpha_beta_search(self, state, depth):

        def min_value(state, alpha, beta, depth):
            
            if state.terminal_test(): 
                return state.utility(self.player_id)
            if depth <= 0 : 
                #return self.heuristic_score(state)
                return self.custom_heuristic(state)
            
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), alpha, beta, depth-1))
                if value <= alpha:
                    return value
                beta = min(beta, value)
            return value
        

        def max_value(state, alpha, beta, depth):

            if state.terminal_test(): 
                return state.utility(self.player_id)
            if depth <= 0 : 
                #return self.heuristic_score(state)
                return self.custom_heuristic(state)

            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), alpha, beta, depth-1))
                if value >= beta:
                    return value
                alpha = max(alpha, value)
            return value

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
        

    def heuristic_score(self, state):
        # A list containing the position of open liberties in the
        # neighborhood of the starting position
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        
        return len(own_liberties) - len(opp_liberties)
        
        
        
    def custom_heuristic(self, state):
        # A list containing the position of open liberties in the
        # neighborhood of the starting position
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        
        
        # states away from edges to be encouraged, so the weight is bigger
        if self.distance(state) >= 2: 
            return 2*len(own_liberties) - len(opp_liberties)
        else: 
            return len(own_liberties) - len(opp_liberties)
        
        return len(own_liberties) - len(opp_liberties)

    
    
    def distance(self, state):
        """ minimum distance to the walls """
        own_loc = state.locs[state.ply_count % 2]
        x, y = own_loc // (_WIDTH + 2), own_loc % (_WIDTH + 2)

        return min(x, _WIDTH + 1 - x, y, _HEIGHT - 1 - y)
    
    

       
