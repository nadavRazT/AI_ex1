import numpy as np

from board import Board
from search import SearchProblem, ucs
import util

EMPTY = -1


class BlokusFillProblem(SearchProblem):
    """
    A one-player Blokus game as a search problem.
    This problem is implemented for you. You should NOT change it!
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        state: Search state
        Returns True if and only if the state is a valid goal state
        """
        return not any(state.pieces[0])

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, 1) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)


#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################
class BlokusCornersProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.corners = [(0, board_w - 1), (board_h - 1, board_w - 1), (board_h - 1, 0)]
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def _cover_corner(self, state, corner):
        if state[corner[0]][corner[1]] == EMPTY:
            return False
        return True

    def is_goal_state(self, state):
        for corner in self.corners:
            if not self._cover_corner(state.state, corner):
                return False
        return True

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        total_cost = 0
        for action in actions:
            total_cost += action.pieces.get_num_tiles()
        return total_cost


class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.targets = targets.copy()
        self.expanded = 0
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def _cover_piece(self, state, piece):
        if state[piece[0]][piece[1]] == -1:
            return False
        return True

    def is_goal_state(self, state):
        for target in self.targets:
            if not self._cover_piece(state.state, target):
                return False
        return True

    def get_successors(self, state):
        """
        state: Search state
        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take
        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        total_cost = 0
        for action in actions:
            total_cost += action.pieces.get_num_tiles()
        return total_cost


def blokus_cover_heuristic(state, problem):
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def calc_piece_val(piece):
    # piece = son[1].piece
    x_list = piece.x
    y_list = piece.y
    n = piece.num_tiles
    w = max(x_list) - min(x_list) + 1
    h = max(y_list) - min(y_list) + 1
    d = np.sqrt(w ^ 2 + h ^ 2)
    return n / d


def order_piece(state, valid_pieces):
    pieces = state.pieces
    piece_list = state.piece_list
    for i in range(len(pieces)):
        if not pieces[0][i]:
            continue
        valid_pieces.append(piece_list.get_piece(i))
    valid_pieces.sort(key=calc_piece_val)
    return


def optimal_distance(xy1, xy2, valid_pieces):
    # small = min(abs(xy1[0] - xy2[0]), abs(xy1[1] - xy2[1]))
    # big = max(abs(xy1[0] - xy2[0]), abs(xy1[1] - xy2[1]))
    # for piece in valid_pieces:
    #     max_x = max(piece.x) - min(piece.x) + 1
    #     max_y = max(piece.y) - min(piece.y) + 1
    #
    #     if (max_x < big and max_y < small) or (max_y < big and max_x < small):
    #         remain_dist = big - max(max_x, max_y)
    #         return piece.get_num_tiles() + remain_dist
    return np.linalg.norm(np.array(xy1) - np.array(xy2))


def blokus_corners_heuristic(state, problem):
    """
    Your heuristic for the BlokusCornersProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.
    """
    board_w = state.board_w
    board_h = state.board_h
    corners = [(board_h - 1, 0), (board_h - 1, board_w - 1), (0, board_w)]
    cost = 0
    valid_pieces = list()
    order_piece(state, valid_pieces)
    for corner in corners:
        min = np.inf
        for y in range(board_h):
            for x in range(board_w):
                if state.check_tile_attached(0, x, y):
                    # d = util.manhattanDistance((x,y), corner)
                    d = optimal_distance((x, y), corner, valid_pieces)
                    if min > d:
                        min = d
        cost += min
    return cost


def blokus_corners_heuristic_test(state, problem):
    sons = problem.get_successors(state)
    if not sons:
        return 0
    min_son = min(sons, key=calc_piece_val)
    return calc_piece_val(min_son)


class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.targets = targets.copy()
        self.expanded = 0
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def blokus_cover_heuristic(state, problem):
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


class ClosestLocationSearch:
    """
    In this problem you have to cover all given positions on the board,
    but the objective is speed, not optimality.
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.targets = targets.copy()
        self.expanded = 0
        self.board_w = board_w
        self.board_h = board_h
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.current_corners = []
        self.piece_list = piece_list
        self.starting_point = starting_point

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def _cover_piece(self, state, piece):
        if state[piece[0]][piece[1]] == EMPTY:
            return False
        return True

    def is_goal_state(self, state):
        for target in self.targets:
            if not self._cover_piece(state.state, target):
                return False
        return True

    def get_successors(self, state):
        """
        state: Search state
        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def _get_corners(self, state):
        corners = []
        for y in range(self.board_h):
            for x in range(self.board_w):
                if state.check_tile_attached(0, x, y):
                    corners.append((x, y))
        return corners

    def _closest_corner_distance(self, target):
        euk_dist_from_target = lambda corner: np.linalg.norm(np.array(target) - np.array(corner))
        return min(self.current_corners, key=euk_dist_from_target)

    def _closest_corner_distance_state(self,state, target):
        corners = self._get_corners(state)
        euk_dist_from_target = lambda corner: np.linalg.norm(np.array(target) - np.array(corner))
        return euk_dist_from_target(min(corners, key=euk_dist_from_target))


    def _is_point_in_board(self, x, y):
        if x >= 0 and x < self.board_w and y >= 0 and y < self.board_h:
            return True
        return False


    def _is_kissing(self, board):
        flag = False
        for target in self.targets:
            curr_x = target[1]
            curr_y = target[0]
            flag = False
            if board.get_position(curr_x, curr_y) != EMPTY:
                return flag

            if self._is_point_in_board(curr_x + 1, curr_y + 1):
                if not board.get_position(curr_x + 1, curr_y + 1):
                    flag = True

            if self._is_point_in_board(curr_x - 1, curr_y + 1):
                if not board.get_position(curr_x - 1, curr_y + 1):
                    flag = True

            if self._is_point_in_board(curr_x + 1, curr_y - 1):
                if not board.get_position(curr_x + 1, curr_y - 1):
                    flag = True

            if self._is_point_in_board(curr_x - 1, curr_y - 1):
                if not board.get_position(curr_x - 1, curr_y - 1):
                    flag = True

        return flag


    def move_to_next_target(self, board, target):
        if board.state[target[0]][target[1]] != EMPTY:
            return []
        actions = []
        while board.state[target[0]][target[1]] == EMPTY:
            neighbors = self.get_successors(board)
            if not neighbors:
                return []
            action = 0
            min_dist = np.inf
            for neighbor in neighbors:
                if self._is_kissing(neighbor[0]):
                    continue
                if neighbor[0].state[target[0]][target[1]] != EMPTY:
                    action = neighbor[1]
                    break

                tent = self._closest_corner_distance_state(neighbor[0], target)
                if tent < min_dist:
                    min_dist = tent
                    action = neighbor[1]
            board.add_move(0, action)
            actions.append(action)
        return actions


    def solve(self):
        """
        This method should return a sequence of actions that covers all target locations on the board.
        This time we trade optimality for speed.
        Therefore, your agent should try and cover one target location at a time. Each time, aiming for the closest uncovered location.
        You may define helpful functions as you wish.

        Probably a good way to start, would be something like this --

        """

        current_state = self.board.__copy__()
        backtrace = []
        while self.targets:
            self.current_corners = self._get_corners(current_state)
            self.targets.sort(key=self._closest_corner_distance)
            target = self.targets.pop()
            actions = self.move_to_next_target(current_state, target)
            # for action in actions:
            #     current_state.add_move(0, action)

            backtrace += actions

        return backtrace


class MiniContestSearch:
    """
    Implement your contest entry here
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.targets = targets.copy()
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def solve(self):
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
