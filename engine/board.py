import chess
import numpy as np
from typing import Optional

PIECE_TO_CHANNEL = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

def board_to_tensor(board: chess.Board) -> np.ndarray:
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            channel = PIECE_TO_CHANNEL[piece.piece_type]
            if piece.color == chess.BLACK:
                channel += 6
            tensor[channel, chess.square_rank(square), chess.square_file(square)] = 1.0

    return tensor

def board_to_tensor_extended(board: chess.Board) -> np.ndarray:
    tensor = np.zeros((19, 8, 8), dtype=np.float32)
    
    tensor[:12] = board_to_tensor(board)

    # Side to move (channel 12)
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0

    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        tensor[17, :, ep_file] = 1.0
    
    halfmove_normalized = min(board.halfmove_clock / 100.0, 1.0)
    tensor[18, :, :] = halfmove_normalized
    
    return tensor

def tensor_to_board(tensor: np.ndarray) -> chess.Board:
    board = chess.Board(fen=None)

    channel_to_piece = {
        0: (chess.PAWN, chess.WHITE),
        1: (chess.KNIGHT, chess.WHITE),
        2: (chess.BISHOP, chess.WHITE),
        3: (chess.ROOK, chess.WHITE),
        4: (chess.QUEEN, chess.WHITE),
        5: (chess.KING, chess.WHITE),
        6: (chess.PAWN, chess.BLACK),
        7: (chess.KNIGHT, chess.BLACK),
        8: (chess.BISHOP, chess.BLACK),
        9: (chess.ROOK, chess.BLACK),
        10: (chess.QUEEN, chess.BLACK),
        11: (chess.KING, chess.BLACK),
    }

    for channel in range(12):
        piece_type, color = channel_to_piece[channel]
        for row in range(8):
            for col in range(8):
                if tensor[channel, row, col] > 0.5:
                    square = chess.square(col, row)
                    piece = chess.Piece(piece_type, color)
                    board.set_piece_at(square, piece)
    
    return board

def flip_board_tensor(tensor: np.ndarray) -> np.ndarray:
    flipped = np.zeros_like(tensor)
    
    for i in range(6):
        flipped[i] = np.flip(tensor[i + 6], axis=0)
        flipped[i + 6] = np.flip(tensor[i], axis=0)
    
    if tensor.shape[0] > 12:
        flipped[12] = 1.0 - tensor[12]
        
        flipped[13] = tensor[15]
        flipped[14] = tensor[16]
        flipped[15] = tensor[13]
        flipped[16] = tensor[14]
        
        if tensor.shape[0] > 17:
            flipped[17] = np.flip(tensor[17], axis=0)
        
        if tensor.shape[0] > 18:
            flipped[18] = tensor[18]
    
    return flipped

def visualize_tensor(tensor: np.ndarray, channel: Optional[int] = None):
    piece_symbols = {
        0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K',
        6: 'p', 7: 'n', 8: 'b', 9: 'r', 10: 'q', 11: 'k'
    }
    
    if channel is not None:
        print(f"Channel {channel}:")
        for row in range(7, -1, -1):
            line = ""
            for col in range(8):
                val = tensor[channel, row, col]
                line += "1 " if val > 0.5 else ". "
            print(f"{row + 1} {line}")
        print("  a b c d e f g h")
    else:
        print("Combined view:")
        for row in range(7, -1, -1):
            line = ""
            for col in range(8):
                piece_found = False
                for ch in range(12):
                    if tensor[ch, row, col] > 0.5:
                        line += piece_symbols[ch] + " "
                        piece_found = True
                        break
                if not piece_found:
                    line += ". "
            print(f"{row + 1} {line}")
        print("  a b c d e f g h")