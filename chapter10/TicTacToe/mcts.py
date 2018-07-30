from TicTacToe import TicTacToe
from copy import deepcopy
from math import log, sqrt
from random import choice as rndchoice
import time

class GameTree:
    def __init__(self, s, par_node=None, pre_action=None):
        self.parent = par_node
        self.pre_action = pre_action
        self.child = []
        self.r = 0
        self.n = 0
        self.state = s
        self.player = MCTS.current_player(s)
        self.uct = float('inf')
        self.result = MCTS.terminal(s)

    def __repr__(self):
        ratio = self.r / (self.n + 1)
        l = [str(e) for e in (self.pre_action, ''.join(self.state), self.r, self.n, str(ratio)[:5], str(self.uct)[:5])]
        return ' '.join(l)

    def update(self, v):
        self.n += 1
        if v == 3:
            self.r += 0.5
        elif v == 3 - self.player:
            self.r += 1


class MCTS:
    def __init__(self, s):
        self.root = GameTree(s)
        self.game = TicTacToe() ## Initialize the game
        self.expand_node(self.root) ## Start the initial node expansion

    def run_mcts(self, board):
        self.__init__(board)
        start_time = time.time()
        iii = 0
        while time.time() - start_time < 2:
            self.mcts_loop()
            iii += 1

    def ai_move(self):
        best_node, best_visits = None, 0
        for n in self.root.child:
            if n.n > best_visits: best_visits, best_node = n.n, n
        return best_node.pre_action

    def mcts_loop(self):
        node = self.node_selection(self.root)
        self.expand_node(node)
        if node.child:
            selected_node = rndchoice(node.child)
        else:
            selected_node = node
        v = self.simulation(deepcopy(selected_node.state))
        self.backpropagation(selected_node, v)

    def node_selection(self, node):
        if node.child:
            imax, vmax = 0, 0
            for i, n in enumerate(node.child):
                n.uct = MCTS.uct(n)
                v = n.uct
                if v > vmax:
                    imax, vmax = i, v
            selected = node.child[imax]
            return self.node_selection(selected)
        else:
            selected = node
            return selected

    def expand_node(self, node):
        if self.terminal(node.state) == 0:
            actions = self.available_move(node.state)
            for a in actions:
                state_after_action = self.action_result(node.state, a)
                node.child.append(GameTree(state_after_action, node, a))

    def simulation(self, s):
        if self.terminal(s) == 0:
            actions = self.available_move(s)
            a = rndchoice(actions)
            s = self.action_result(s, a)
            return s
        else:
            return self.terminal(s)

    def backpropagation(self, node, v):
        node.update(v)
        if node.parent:
            self.backpropagation(node.parent, v)

    @staticmethod
    def terminal(s):
        for wc in TicTacToe().winning_cases:
            if s[wc[0]] != '_' and \
                    s[wc[0]] == s[wc[1]] and \
                    s[wc[1]] == s[wc[2]]:
                if s[wc[0]] == 'X':
                    return 1
                else:
                    return 2
        if '_' not in s:
            return 3
        else:
            return 0

    @staticmethod
    def available_move(s):
        l = []
        for i in range(9):
            if s[i] == '_': l.append(i)
        return l

    @staticmethod
    def action_result(s, a):
        p = MCTS.current_player(s)
        new_s = deepcopy(s)
        new_s[a] = 'X' if p == 1 else 'O'
        return new_s

    @staticmethod
    def current_player(s):
        n = s.count('_')
        if n % 2 == 1:
            return 1
        else:
            return 2

    @staticmethod
    def uct(node):
        v = (node.r / (node.n + 1e-12)) + sqrt(2 * log(node.parent.n + 1) / (node.n + 1e-12))
        return v


if __name__ == '__main__':
    game = TicTacToe()
    ai = MCTS(game.board)
    while game.result == 0:
        game.display_board()
        ai.run_mcts(board=game.board)
        game.switch_player(ai.ai_move())
        game.check_result()
    game.display_board()
    if game.result == 3:
        print('The game has ended in a draw')
    else:
        print(f'Player {game.result} has won the game')
