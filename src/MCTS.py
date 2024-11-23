import chess
import numpy as np


class Node():
    def __init__(
            self,
            board,
            parent=[],
            is_white=True,
            l_castle=True,
            s_castle=True,
              ):
        self.reward = 0.0
        self.board = board
        self.children = []
        self.parent = parent
        self.visits = 1
        self.is_white = is_white
        self.s_castle = s_castle
        self.l_castle = l_castle

    def game_over(self):
        if self.board.is_checkmate():
            return True
        elif self.board.is_repetition():
            return True
        elif self.board.is_stalemate():
            return True
        elif self.board.is_seventyfive_moves():
            return True
        elif self.board.is_fifty_moves():
            return True
        elif self.board.is_insufficient_material():
            return True
        else:
            return False


class chess_MCTS():
    def __init__(self):
        self.board = chess.Board()
        self.moves = list(self.board.legal_moves)
        board = self.board.copy()
        self.initial_node = Node(board)
        code = board.unicode() + 'w'
        self.states = set(code)
        self.dict_nodes = {board.unicode(): self.initial_node}

    def add_children(self, node):
        moves = list(node.board.legal_moves)
        for move in moves:
            board = node.board.copy()
            is_white = node.is_white
            board.push(move)
            code = board.unicode() + is_white * "w"
            if code in self.states:
                c_node = self.dict_nodes[code]
                c_node.parent.append(node)
                node.children.append(c_node)
            else:
                new_node = Node(board, parent=[node], is_white=1-is_white)
                self.states.add(code)
                self.dict_nodes[code] = new_node
                node.children.append(new_node)

    def __call__(self, num_iteration, init_node=None):
        for _ in range(num_iteration):
            if init_node is not None:
                node = init_node
            else:
                node = self.initial_node
            w_nodes = [node.board.unicode() + "w"]
            b_nodes = []
            i = 0
            while not node.game_over():
                if node.children == []:
                    self.add_children(node)
                crit = [(c.reward/c.visits) +
                        np.sqrt(np.log(node.visits)/c.visits)
                        for c in node.children]
                node.visits += 1
                node = node.children[np.argmax(crit)]
                if node.is_white:
                    w_nodes.append(node.board.unicode() + "w")
                else:
                    b_nodes.append(node.board.unicode())
                i += 1

            if node.board.is_checkmate():
                w_reward = 1 + node.is_white * (-2)
                b_reward = - w_reward
                for code in w_nodes:
                    self.dict_nodes[code].reward += w_reward
                for code in b_nodes:
                    self.dict_nodes[code].reward += b_reward
            print(_, node.board.is_checkmate(), i)
        return node
