import torch
import re
import numpy as np
import chess


def from_board_to_matrix(unicode):
    Board = torch.zeros(12, 64)
    board = chess.Board()
    pieces = list(set(board.unicode()))
    pieces.remove('\n')
    pieces.remove(' ')
    pieces.remove('⭘')
    b = b = re.sub(" ", "", unicode)
    b = re.sub("\n", "", b)
    b_array = np.array(list(b))
    for index, piece in enumerate(pieces):
        ii = np.where(b_array == piece)[0]
        ii = torch.from_numpy(ii)
        Board[ii, index] = torch.tensor(1, dtype=torch.float32)
    return Board.view(-1, 8, 8)
