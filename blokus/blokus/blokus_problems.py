import numpy as np

from board import Board
from search import SearchProblem, ucs
import util
import search

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


def calc_piece_size(piece):
    x_list = piece.x
    y_list = piece.y
    n = piece.num_tiles
    w = max(x_list) - min(x_list) + 1
    h = max(y_list) - min(y_list) + 1
    return max(w, h), min(w, h)


def get_corners(state):
    corners = []
    for y in range(state.board_h):
        for x in range(state.board_w):
            if state.check_tile_legal(0, x, y) and state.check_tile_attached(0, x, y):
                corners.append((x, y))
    return corners


def is_point_in_board(state, x, y):
    if x >= 0 and x < state.board_w and y >= 0 and y < state.board_h:
        return True
    return False


def is_kissing(board, targets):
    flag = False
    for target in targets:
        curr_x = target[1]
        curr_y = target[0]
        flag = False
        if board.get_position(curr_x, curr_y) != EMPTY:
            return flag

        if is_point_in_board(board, curr_x + 1, curr_y + 1):
            if not board.get_position(curr_x + 1, curr_y + 1):
                flag = True

        if is_point_in_board(board, curr_x - 1, curr_y + 1):
            if not board.get_position(curr_x - 1, curr_y + 1):
                flag = True

        if is_point_in_board(board, curr_x + 1, curr_y - 1):
            if not board.get_position(curr_x + 1, curr_y - 1):
                flag = True

        if is_point_in_board(board, curr_x - 1, curr_y - 1):
            if not board.get_position(curr_x - 1, curr_y - 1):
                flag = True

    return flag


def sort_corners(state_corners, target):
    euk_dist_from_target = lambda corner: np.linalg.norm(np.array(target) - np.array(corner))
    state_corners.sort(key=euk_dist_from_target)
    return


def check_rect_valid(state, target, c_corner):
    x_dif = abs(c_corner[1] - target[1])
    y_dif = abs(c_corner[0] - target[0])
    big = max(x_dif, y_dif) + 1
    small = min(y_dif, x_dif) + 1
    pieces = state.pieces
    piece_list = state.piece_list
    for i in range(len(pieces)):
        if not pieces[0][i]:
            continue
        p_big, p_small = calc_piece_size(piece_list.get_piece(i))
        if big >= p_big and small >= p_small:
            return True
    return False


def get_min_piece_size(state):
    pieces = state.pieces
    piece_list = state.piece_list
    min_size = np.inf
    for i in range(len(pieces)):
        if not pieces[0][i]:
            continue
        piece_size = piece_list.get_piece(i).get_num_tiles()
        if piece_size < min_size:
            min_size = piece_size
    return min_size


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
    corners = [(board_h - 1, 0), (board_h - 1, board_w - 1), (0, board_w - 1)]
    if problem.is_goal_state(state):
        return 0
    if is_kissing(state, corners):
        return np.inf
    max_min_dist = 0
    min_piece_size = get_min_piece_size(state)
    cost = 0
    n_covered = 1
    for corner in corners:
        if state.get_position(corner[1], corner[0]) != EMPTY:
            n_covered += 1
            continue
        euk_dist_from_target = lambda target: np.linalg.norm(np.array(target) - np.array(corner))
        state_corners = get_corners(state)
        sort_corners(state_corners, corner)
        if not state_corners:
            return np.inf
        closest_dist = euk_dist_from_target(state_corners[0])
        min_dist = min_piece_size + closest_dist
        while state_corners:
            c_corner = state_corners[0]
            rect_valid = check_rect_valid(state, corner, c_corner)
            if not rect_valid:
                state_corners.pop(0)
                continue
            min_dist = euk_dist_from_target(c_corner)
            break
        if max_min_dist < min_dist:
            max_min_dist = min_dist
    return max_min_dist + 3 - n_covered


def blokus_cover_heuristic(state, problem):
    # targets = problem.targets
    #
    # if problem.is_goal_state(state):
    #     return 0
    # if is_kissing(state, targets):
    #     return np.inf
    #
    # min_dist = get_min_piece_size(state)
    # cost = 0
    # n_covered = 1
    # for target in targets:
    #     if state.get_position(target[1], target[0]) != EMPTY:
    #         n_covered += 1
    #         continue
    #
    #     state_corners = get_corners(state)
    #     sort_corners(state_corners, target)
    #     while state_corners:
    #         c_corner = state_corners[0]
    #         # rect_valid = check_rect_valid(state, target, c_corner)
    #         # if not rect_valid:
    #         #     state_corners.pop(0)
    #         #     continue
    #         euk_dist_from_corner = lambda corner: np.linalg.norm(np.array(target) - np.array(corner))
    #         min_dist = euk_dist_from_corner(c_corner)
    #         break
    #     cost += min_dist
    # board_w = state.board_w
    # board_h = state.board_h
    # corners = [(board_h - 1, 0), (board_h - 1, board_w - 1), (0, board_w - 1)]
    targets = problem.targets
    if problem.is_goal_state(state):
        return 0
    if is_kissing(state, targets):
        return np.inf
    n_covered = 1
    max_min_dist = 0
    min_piece_size = get_min_piece_size(state)
    for target in targets:
        if state.get_position(target[1], target[0]) != EMPTY:
            n_covered += 1
            continue
        state_corners = get_corners(state)
        sort_corners(state_corners, target)
        euk_dist_from_target = lambda corner: np.linalg.norm(np.array(target) - np.array(corner))
        closest_dist = euk_dist_from_target(state_corners[0])
        min_dist = min_piece_size + closest_dist
        while state_corners:
            c_corner = state_corners[0]
            rect_valid = check_rect_valid(state, target, c_corner)
            if not rect_valid:
                state_corners.pop(0)
                continue
            min_dist = euk_dist_from_target(c_corner)
            break
        if max_min_dist < min_dist:
            max_min_dist = min_dist
    return max_min_dist + len(targets) - n_covered


class ClosestLocationSearch:
    """
    In this problem you have to cover all given positions on the board,
    but the objective is speed, not optimality.
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.targets = targets.copy()
        self.orig_targets = targets.copy()
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
        val = min(self.current_corners, key=euk_dist_from_target)
        return euk_dist_from_target(val)

    def _closest_corner_distance_state(self, state, target):
        corners = self._get_corners(state)
        euk_dist_from_target = lambda corner: np.linalg.norm(np.array(target) - np.array(corner))
        return euk_dist_from_target(min(corners, key=euk_dist_from_target))

    def _is_point_in_board(self, x, y):
        if x >= 0 and x < self.board_w and y >= 0 and y < self.board_h:
            return True
        return False

    def _is_kissing(self, board, targets):
        flag = False
        for target in targets:
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

    def move_to_next_target(self):
        actions = search.uniform_cost_search(self)
        # if board.state[target[0]][target[1]] != EMPTY:
        #     return []
        # actions = []
        # while board.state[target[0]][target[1]] == EMPTY:
        #     neighbors = self.get_successors(board)
        #     if not neighbors:
        #         return []
        #     action = 0
        #     min_dist = np.inf
        #     for neighbor in neighbors:
        #         all_targets = self.targets + [target]
        #         if self._is_kissing(neighbor[0], all_targets):
        #             continue
        #         if neighbor[0].state[target[0]][target[1]] != EMPTY:
        #             action = neighbor[1]
        #             break
        #
        #         tent = self._closest_corner_distance_state(neighbor[0], target)
        #         if tent < min_dist:
        #             min_dist = tent
        #             action = neighbor[1]
        #     board.add_move(0, action)
        #     actions.append(action)

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

        while self.orig_targets:
            self.current_corners = self._get_corners(current_state)
            self.orig_targets.sort(key=self._closest_corner_distance)
            self.targets = [self.orig_targets.pop()]
            actions = self.move_to_next_target()
            for action in actions:
                self.board.add_move(0, action)
            backtrace += actions
        self.board = current_state
        return backtrace


class MiniContestSearch:
    """
    Implement your contest entry here
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.targets = targets.copy()
        self.orig_targets = targets.copy()
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

    def solve(self):
        back_trace = search.a_star_search(self, blokus_cover_heuristic)
        return back_trace
