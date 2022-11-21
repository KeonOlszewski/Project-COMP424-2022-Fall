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
        valid_moves = self.get_all_moves(chess_board, my_pos, adv_pos, max_step)
        chosen_move = valid_moves[-1] #TODO make an algorithm that picks moves better
        return (chosen_move[0],chosen_move[1]), chosen_move[2]

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
    
    def get_all_moves(self, chess_board, og_pos, adv_pos, max_step):
        moves = [og_pos]
        new_moves = [(og_pos[0],og_pos[1],0)] #starting position with 0 steps so far
        while len(new_moves) > 0:
            move = new_moves[0]
            new_moves = new_moves[1:]
            new = self.get_moves_from_position(chess_board, move, adv_pos, max_step, moves, new_moves)
            for move in new:
                moves.append((move[0], move[1]))
            new_moves += new
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