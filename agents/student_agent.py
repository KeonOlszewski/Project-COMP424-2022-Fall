# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import random
from copy import deepcopy
import numpy as np
import time


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.autoplay = True
        self.root = None
        self.turn = 1
        self.previous_board = None
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    # def step(self, chess_board, my_pos, adv_pos, max_step):
    #     valid_moves = self.get_valid_moves(chess_board, my_pos, adv_pos, max_step)
    #     chosen_move =  self.get_best_move(chess_board, valid_moves, my_pos, adv_pos,)
    #     return (chosen_move[0],chosen_move[1]), chosen_move[2]

    def step(self, chess_board, my_pos, adv_pos, max_step):
        start = time.time()

        valid_moves = self.get_valid_moves(chess_board, my_pos, adv_pos, max_step)
        winrates = [0 for i in range(len(valid_moves))]

        #make nodes for every valid move (might be overkill)
        self.root = self.MCNode(chess_board, my_pos, None, adv_pos, max_step, True, None)
        for move in valid_moves:
            new_board = deepcopy(chess_board)
            self.set_barrier(move[0], move[1], move[2], new_board)
            new_node = self.MCNode(new_board, (move[0], move[1]), move[2], adv_pos, max_step, False, self.root)
            self.root.children.append(new_node)
        if self.turn == 1:
            while (time.time() - start) < 0: #CHANGE TO 28 FOR FINAL VERSION
                node = self.tree_policy()
                node.simulate()
            self.turn = 2
        num = 0
        while (time.time() - start) < 1.95:
            node = self.tree_policy()
            node.simulate()
            num += 1
        for i in range(len(self.root.children)):
            winrates[i] = self.root.children[i].get_winrate()
        max_winrate_index= np.argmax(winrates)
        champion_node = self.root.children[max_winrate_index]
        moves = []
        for i in range(len(self.root.children)):
            moves.append(self.root.children[i].my_pos)
        return champion_node.get_mypos(), champion_node.get_barrier()

    def tree_policy(self):
        for node in self.root.children:
            if node.winrate > 0.9 and node.num_games < 30:
                return node
            elif node.num_games < 5:
                return node
        return self.root.children[random.randrange(len(self.root.children))]

    '''
    Currently for each valid moves gets the score of the board if that move was made.
    If that move keeps the score the same as current score, play that move.
    Otherwise, append it to the score list and check next move.
    '''
    def get_best_move(self, chess_board, valid_moves, my_pos, adv_pos):
        scores = []
        cur_score = self.get_score(chess_board, my_pos)
        for i in range(len(valid_moves)):
            move = valid_moves[i]
            chess_board[move[0], move[1], move[2]] = True
            score = self.get_score(chess_board, (move[0],move[1]))
            chess_board[move[0], move[1], move[2]] = False
            if score == cur_score:
                return valid_moves[i]
            scores.append(score)
        return valid_moves[scores.index(max(scores))]

    '''
    Score is identical to number of reachable positions given a max step size 2*board_size (full board).
    Need to set the adv position to your position so the adversary doesn't block any positions.
    '''
    def get_score(self, chess_board, pos):
        return len(self.get_valid_positions(chess_board, pos, pos, len(chess_board)*len(chess_board)))

    '''
    Given a position on the board, returns a list of adjacent open positions.
    Takes into account max step size, adv position, barriers and the current valid moves list 
    (so as not to include duplicates)
    '''
    def get_moves_from_position(self, chess_board, pos, adv_pos, max_step, moves, steps):
        #Up, Right, Down, Left
        #(-1, 0), (0, 1), (1, 0), (0, -1)
        x = pos[0]
        y = pos[1]
        steps = pos[2]
        new_moves = []
        if steps + 1 > max_step:
            return []
        if not chess_board[x][y][self.dir_map["u"]] and (x-1, y) not in moves and not (x-1, y) == adv_pos:
            new_moves.append((x-1, y, steps+1))
        if not chess_board[x][y][self.dir_map["r"]] and (x, y+1) not in moves and not (x, y+1) == adv_pos:
            new_moves.append((x, y+1, steps+1))
        if not chess_board[x][y][self.dir_map["d"]] and (x+1, y) not in moves and not (x+1, y) == adv_pos:
            new_moves.append((x+1, y, steps+1))
        if not chess_board[x][y][self.dir_map["l"]] and (x, y-1) not in moves and not (x, y-1) == adv_pos:
            new_moves.append((x, y-1, steps+1))
        return new_moves

    '''
    Continuously calls the get_moves_from_position function on each new position found until all
    reachable positions have been found
    '''
    def get_valid_positions(self, chess_board, og_pos, adv_pos, max_step):
        moves = [og_pos]
        new_moves = [(og_pos[0],og_pos[1],0)] #starting position with 0 steps so far
        while len(new_moves) > 0:
            move = new_moves[0]
            new_moves = new_moves[1:]
            new = self.get_moves_from_position(chess_board, move, adv_pos, max_step, moves, new_moves)
            for move in new:
                moves.append((move[0], move[1]))
            new_moves += new
        return moves
    
    '''
    Uses get_valid_positions to get a list of all reachable positions.
    Then determines from each position what barriers can be placed and creates a list of all
    position/barrier combinations that make up a move.
    '''
    def get_valid_moves(self, chess_board, og_pos, adv_pos, max_step):
        moves = self.get_valid_positions(chess_board, og_pos, adv_pos, max_step)
        moves_with_barrier = []
        for move in moves:
            if not chess_board[move[0]][move[1]][self.dir_map["u"]]:
                moves_with_barrier.append((move[0],move[1],self.dir_map["u"]))
            if not chess_board[move[0]][move[1]][self.dir_map["d"]]:
                moves_with_barrier.append((move[0],move[1],self.dir_map["d"]))
            if not chess_board[move[0]][move[1]][self.dir_map["l"]]:
                moves_with_barrier.append((move[0],move[1],self.dir_map["l"]))
            if not chess_board[move[0]][move[1]][self.dir_map["r"]]:
                moves_with_barrier.append((move[0],move[1],self.dir_map["r"]))
        return moves_with_barrier
    
    def set_barrier(self, r, c, dir, chess_board):

      moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
      opposites = {0: 2, 1: 3, 2: 0, 3: 1}

      # Set the barrier to True
      chess_board[r, c, dir] = True
      # Set the opposite barrier to True
      move = moves[dir]
      chess_board[r + move[0], c + move[1], opposites[dir]] = True
      return chess_board
             
    class MCNode:  
        def __init__(self, chess_board, my_pos, barrier, adv_pos, max_step, my_turn, root):
            self.parent = root
            self.children = []
            self.chess_board = chess_board 
            self.my_turn = my_turn
            self.my_pos = my_pos
            self.barrier = barrier
            self.adv_pos = adv_pos 
            self.max_step = max_step
            self.num_games = 0
            self.num_wins = 0
            self.winrate = 0
            self.dir_map = {
                "u": 0,
                "r": 1,
                "d": 2,
                "l": 3,
            }

        def default_policy(self, my_pos, adv_pos, chess_board):
            return self.randmov(my_pos, adv_pos, chess_board)
            valid_moves = self.get_valid_moves(chess_board, my_pos, adv_pos, self.max_step)
            for move in valid_moves:
                board = deepcopy(chess_board)
                self.set_barrier(move[0],move[1],move[2],board)
                if self.get_score(board, (move[0],move[1])) > self.get_score(board, adv_pos):
                    return (move[0],move[1]),move[2]
            return self.randmov(my_pos, adv_pos, chess_board)
        
        '''
        randomly makes moves for us and the opponent until game is over
        updates winrate and number of games 
        '''
        def simulate(self):

            my_turn = self.my_turn # to keep track of turns initially true
            check_endgame = None
            my_pos, adv_pos = self.my_pos, self.adv_pos #positions of players
            chess_board = deepcopy(self.chess_board) #wall positions
            self.num_games = self.num_games = self.num_games + 1 #keep track of games
            while True:
                check_endgame = self.check_endgame(chess_board, my_pos, adv_pos)
                if check_endgame is not None and check_endgame[0]: #check_endgame[0] is true if the game is over
                    break
                elif my_turn:
                    (x, y), dir = self.default_policy(my_pos, adv_pos, chess_board)
                    my_pos = (x, y)
                else:
                    (x, y), dir = self.default_policy(adv_pos, my_pos, chess_board)
                    adv_pos = (x, y)
                if dir is None:
                    if my_turn:
                        self.winrate = self.num_wins / self.num_games
                    else:
                        self.num_wins += 1
                        self.winrate = self.num_wins / self.num_games
                    return
                my_turn = not my_turn # change turn
                self.set_barrier(x, y, dir, chess_board)
            if(check_endgame[1] > check_endgame[2]): #check if we won and update wins
                self.num_wins = self.num_wins + 1
            self.winrate = self.num_wins/self.num_games #update winrate
        
        '''
        fetch winrate of node
        '''
        def get_winrate(self):
            return self.winrate
        
        '''
        fetch position of move
        '''
        def get_mypos(self):
            return self.my_pos
        
        '''
        fetch barrier placement of move
        '''
        def get_barrier(self):
            return self.barrier
        
        def randmov(self, my_pos, adv_pos, chess_board):
            """
            Randomly walk to the next position in the board.
            from world.py
            """
            moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
            ori_pos = deepcopy(my_pos)
            steps = np.random.randint(0, self.max_step + 1)
            # Random Walk
            for _ in range(steps):
                r, c = my_pos
                dir = np.random.randint(0, 4)
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c)

                # Special Case enclosed by Adversary
                k = 0
                while chess_board[r, c, dir] or my_pos == adv_pos:
                    k += 1
                    if k > 300:
                        break
                    dir = np.random.randint(0, 4)
                    m_r, m_c = moves[dir]
                    my_pos = (r + m_r, c + m_c)

                if k > 300:
                    my_pos = ori_pos
                    break

            # Put Barrier
            dir = np.random.randint(0, 4)
            r, c = my_pos
            k = 0
            while chess_board[r, c, dir]:
                k += 1
                dir = np.random.randint(0, 4)
                if k > 10:
                    dir = None
                    break
            return my_pos, dir
        

        def check_endgame(self, chess_board, my_pos, adv_pos):
            """
            Check if the game ends and compute the current score of the agents.

            from world.py
            -------
            is_endgame : bool
                Whether the game ends.
            player_1_score : int
                The score of player 1.
            player_2_score : int
                The score of player 2.
            """
            moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
            # Union-Find
            father = dict()
            for r in range(np.shape(chess_board)[0]):
                for c in range(np.shape(chess_board)[0]):
                    father[(r, c)] = (r, c)

            def find(pos):
                if father[pos] != pos:
                    father[pos] = find(father[pos])
                return father[pos]

            def union(pos1, pos2):
                father[pos1] = pos2

            for r in range(np.shape(chess_board)[0]):
                for c in range(np.shape(chess_board)[0]):
                    for dir, move in enumerate(moves[1:3]):  # Only check down and right
                        if self.chess_board[r, c, dir + 1]:
                            continue
                        pos_a = find((r, c))
                        pos_b = find((r + move[0], c + move[1]))
                        if pos_a != pos_b:
                            union(pos_a, pos_b)

            for r in range(np.shape(chess_board)[0]):
                for c in range(np.shape(chess_board)[0]):
                    find((r, c))
            p0_r = find(tuple(my_pos))
            p1_r = find(tuple(adv_pos))
            p0_score = list(father.values()).count(p0_r)
            p1_score = list(father.values()).count(p1_r)
            if p0_r == p1_r:
                return False, p0_score, p1_score
            return True, p0_score, p1_score

        def set_barrier(self, r, c, dir, chess_board):
            moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
            opposites = {0: 2, 1: 3, 2: 0, 3: 1}

            # Set the barrier to True
            chess_board[r, c, dir] = True
            # Set the opposite barrier to True
            move = moves[dir]
            chess_board[r + move[0], c + move[1], opposites[dir]] = True
            return chess_board
        
        '''
        Given a position on the board, returns a list of adjacent open positions.
        Takes into account max step size, adv position, barriers and the current valid moves list 
        (so as not to include duplicates)
        '''
        def get_moves_from_position(self, chess_board, pos, adv_pos, max_step, moves, steps):
            #Up, Right, Down, Left
            #(-1, 0), (0, 1), (1, 0), (0, -1)
            x = pos[0]
            y = pos[1]
            steps = pos[2]
            new_moves = []
            if steps + 1 > max_step:
                return []
            if not chess_board[x][y][self.dir_map["u"]] and (x-1, y) not in moves and not (x-1, y) == adv_pos:
                new_moves.append((x-1, y, steps+1))
            if not chess_board[x][y][self.dir_map["r"]] and (x, y+1) not in moves and not (x, y+1) == adv_pos:
                new_moves.append((x, y+1, steps+1))
            if not chess_board[x][y][self.dir_map["d"]] and (x+1, y) not in moves and not (x+1, y) == adv_pos:
                new_moves.append((x+1, y, steps+1))
            if not chess_board[x][y][self.dir_map["l"]] and (x, y-1) not in moves and not (x, y-1) == adv_pos:
                new_moves.append((x, y-1, steps+1))
            return new_moves

        '''
        Continuously calls the get_moves_from_position function on each new position found until all
        reachable positions have been found
        '''
        def get_valid_positions(self, chess_board, og_pos, adv_pos, max_step):
            moves = [og_pos]
            new_moves = [(og_pos[0],og_pos[1],0)] #starting position with 0 steps so far
            while len(new_moves) > 0:
                move = new_moves[0]
                new_moves = new_moves[1:]
                new = self.get_moves_from_position(chess_board, move, adv_pos, max_step, moves, new_moves)
                for move in new:
                    moves.append((move[0], move[1]))
                new_moves += new
            return moves
        
        '''
        Uses get_valid_positions to get a list of all reachable positions.
        Then determines from each position what barriers can be placed and creates a list of all
        position/barrier combinations that make up a move.
        '''
        def get_valid_moves(self, chess_board, og_pos, adv_pos, max_step):
            moves = self.get_valid_positions(chess_board, og_pos, adv_pos, max_step)
            moves_with_barrier = []
            for move in moves:
                if not chess_board[move[0]][move[1]][self.dir_map["u"]]:
                    moves_with_barrier.append((move[0],move[1],self.dir_map["u"]))
                if not chess_board[move[0]][move[1]][self.dir_map["d"]]:
                    moves_with_barrier.append((move[0],move[1],self.dir_map["d"]))
                if not chess_board[move[0]][move[1]][self.dir_map["l"]]:
                    moves_with_barrier.append((move[0],move[1],self.dir_map["l"]))
                if not chess_board[move[0]][move[1]][self.dir_map["r"]]:
                    moves_with_barrier.append((move[0],move[1],self.dir_map["r"]))
            return moves_with_barrier
        
        '''
        Score is identical to number of reachable positions given a max step size 2*board_size (full board).
        Need to set the adv position to your position so the adversary doesn't block any positions.
        '''
        def get_score(self, chess_board, pos):
            return len(self.get_valid_positions(chess_board, pos, pos, 2*len(chess_board)))
        
    #    class MCSearchTree:  
    #     def __init__(self, chess_board, my_pos, adv_pos, max_step, my_turn): 
    #         self.my_pos = my_pos
    #         self.chess_board = chess_board 
    #         self.adv_pos = adv_pos 
    #         self.max_step = max_step
    #         self.nodes = []
    #         self.dir_map = {
    #             "u": 0,
    #             "r": 1,
    #             "d": 2,
    #             "l": 3,
    #         }
        
    #     '''COPY
    #     Uses get_valid_positions to get a list of all reachable positions.
    #     Then determines from each position what barriers can be placed and creates a list of all
    #     position/barrier combinations that make up a move.
    #     '''
    #     def get_valid_moves(self, chess_board, og_pos, adv_pos, max_step):
    #         moves = self.get_valid_positions(chess_board, og_pos, adv_pos, max_step)
    #         moves_with_barrier = []
    #         for move in moves:
    #             if not chess_board[move[0]][move[1]][self.dir_map["u"]]:
    #                 moves_with_barrier.append((move[0],move[1],self.dir_map["u"]))
    #             if not chess_board[move[0]][move[1]][self.dir_map["d"]]:
    #                 moves_with_barrier.append((move[0],move[1],self.dir_map["d"]))
    #             if not chess_board[move[0]][move[1]][self.dir_map["l"]]:
    #                 moves_with_barrier.append((move[0],move[1],self.dir_map["l"]))
    #             if not chess_board[move[0]][move[1]][self.dir_map["r"]]:
    #                 moves_with_barrier.append((move[0],move[1],self.dir_map["r"]))
    #         return moves_with_barrier
        
    #     def search(self):
    #         start = time.time()
    #         #maybe check for game over here
    #         valid_moves = self.get_valid_moves(self, self.chess_board, self.my_pos, self.adv_pos, self.max_step)
    #         winrates =  np.zeros(len(valid_moves), dtype = int)

    #         for moves in valid_moves:
    #             move =  valid_moves.pop(random.randrange(len(valid_moves)))
    #             new_board = deepcopy(self.chess_board)
    #             self.set_barrier(self, move[0], move[1], move[2], new_board)
    #             new_node = self.MCNode(new_board, (move[0], move[1]), move[2], self.adv_pos, self.max_step)
    #             self.nodes.append(new_node)
    #         while (time.time() - start) < self.duration:
    #             for i in range(len(self.nodes)):
    #                 self.nodes[i].simulate()
    #                 winrates[i] = self.nodes[i].get_winrate()
            
    #         max_winrate= max(winrates)
    #         champion_node = self.nodes[winrates.index(max_winrate)]
    #         return champion_node.get_mypos(), champion_node.get_barrier()
