# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys


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
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        valid_moves = self.get_valid_moves(chess_board, my_pos, adv_pos, max_step)
        chosen_move =  self.get_best_move(chess_board, valid_moves, my_pos, adv_pos,)
        return (chosen_move[0],chosen_move[1]), chosen_move[2]

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
        return len(self.get_valid_positions(chess_board, pos, pos, 2*len(chess_board)))

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