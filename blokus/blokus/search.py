"""
In search.py, you will implement generic search algorithms
"""

import util
from displays import GuiDisplay
import time


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def is_goal_state(self, state):
        """
        state: Search state

        Returns True if and only if the state is a valid goal state
        """
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
        util.raiseNotDefined()

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def draw_b(board, display):
    # todo delete
    dots = [(board.board_h - 1, board.board_w - 1), (0, board.board_w - 1), (board.board_h - 1, 0)]
    display.draw_board(board, dots=dots)
    input("press Enter: ")


def dfs_helper(node, visited, problem, stack, sol):
    board = node
    for son in reversed(problem.get_successors(node)):
        stack.append(son)
        sol.append(son[1])
        if problem.is_goal_state(son[0]):
            return True, sol
        # check legal
        state = son[0]
        if son is not None and state not in visited:
            visited.append(state)
            sol_flag, sol = dfs_helper(son[0], visited, problem, stack, sol)
            if sol_flag:
                return True, sol
        stack.pop()
        sol.pop()
    return False, sol


def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))
    """

    start_state = problem.get_start_state()
    stack = []
    visited = [start_state]
    # display = GuiDisplay(problem.board.board_w, problem.board.board_h, title='Intro to AI -- 67842 -- Ex1')
    sol = []
    sol_flag, sol = dfs_helper(start_state, visited, problem, stack, sol)
    if sol_flag:
        return sol
    return None


def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def uniform_cost_search(problem):
    """
    Search the node of least total cost first.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def a_star_search(problem, heuristic=null_heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
