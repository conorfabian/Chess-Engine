"""Tests for board encoding functions."""

import chess
import numpy as np
import pytest

from engine.board import (
    board_to_tensor,
    board_to_tensor_extended,
    tensor_to_board,
    flip_board_tensor,
    visualize_tensor,
)


class TestBoardToTensor:
    """Tests for basic board encoding."""
    
    def test_output_shape(self):
        """Tensor should have shape (12, 8, 8)."""
        board = chess.Board()
        tensor = board_to_tensor(board)
        assert tensor.shape == (12, 8, 8)
    
    def test_output_dtype(self):
        """Tensor should be float32."""
        board = chess.Board()
        tensor = board_to_tensor(board)
        assert tensor.dtype == np.float32
    
    def test_starting_position_pawns(self):
        """White pawns on rank 2, black pawns on rank 7."""
        board = chess.Board()
        tensor = board_to_tensor(board)
        
        # White pawns (channel 0) on rank 2 (index 1)
        assert tensor[0, 1, :].sum() == 8
        assert tensor[0, 1, :].min() == 1.0
        
        # Black pawns (channel 6) on rank 7 (index 6)
        assert tensor[6, 6, :].sum() == 8
        assert tensor[6, 6, :].min() == 1.0
    
    def test_starting_position_knights(self):
        """Knights in correct starting positions."""
        board = chess.Board()
        tensor = board_to_tensor(board)
        
        # White knights (channel 1) on b1 and g1
        assert tensor[1, 0, 1] == 1.0  # b1
        assert tensor[1, 0, 6] == 1.0  # g1
        assert tensor[1].sum() == 2
        
        # Black knights (channel 7) on b8 and g8
        assert tensor[7, 7, 1] == 1.0  # b8
        assert tensor[7, 7, 6] == 1.0  # g8
        assert tensor[7].sum() == 2
    
    def test_empty_board(self):
        """Empty board should have all zeros."""
        board = chess.Board(fen=None)
        tensor = board_to_tensor(board)
        assert tensor.sum() == 0
    
    def test_single_piece(self):
        """Single piece encoding."""
        board = chess.Board(fen=None)
        board.set_piece_at(chess.E4, chess.Piece(chess.QUEEN, chess.WHITE))
        tensor = board_to_tensor(board)
        
        # Only white queen channel should have a 1
        assert tensor[4, 3, 4] == 1.0  # e4 = rank 4 (idx 3), file e (idx 4)
        assert tensor.sum() == 1.0
    
    def test_after_moves(self):
        """Encoding updates correctly after moves."""
        board = chess.Board()
        board.push_san("e4")
        tensor = board_to_tensor(board)
        
        # Pawn moved from e2 to e4
        assert tensor[0, 1, 4] == 0.0  # e2 now empty
        assert tensor[0, 3, 4] == 1.0  # e4 has pawn


class TestBoardToTensorExtended:
    """Tests for extended board encoding with game state."""
    
    def test_output_shape(self):
        """Extended tensor should have 19 channels."""
        board = chess.Board()
        tensor = board_to_tensor_extended(board)
        assert tensor.shape == (19, 8, 8)
    
    def test_side_to_move_white(self):
        """Channel 12 all 1s when white to move."""
        board = chess.Board()
        tensor = board_to_tensor_extended(board)
        assert tensor[12].sum() == 64  # All squares = 1
    
    def test_side_to_move_black(self):
        """Channel 12 all 0s when black to move."""
        board = chess.Board()
        board.push_san("e4")
        tensor = board_to_tensor_extended(board)
        assert tensor[12].sum() == 0
    
    def test_castling_rights_start(self):
        """All castling rights available at start."""
        board = chess.Board()
        tensor = board_to_tensor_extended(board)
        
        assert tensor[13].sum() == 64  # White kingside
        assert tensor[14].sum() == 64  # White queenside
        assert tensor[15].sum() == 64  # Black kingside
        assert tensor[16].sum() == 64  # Black queenside
    
    def test_castling_rights_lost(self):
        """Castling rights update when lost."""
        board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1")
        tensor = board_to_tensor_extended(board)
        
        assert tensor[13].sum() == 0   # White kingside lost
        assert tensor[14].sum() == 64  # White queenside available
    
    def test_en_passant(self):
        """En passant square encoded correctly."""
        board = chess.Board()
        board.push_san("e4")
        tensor = board_to_tensor_extended(board)
        
        # En passant on e3 = file e (index 4)
        assert tensor[17, :, 4].sum() == 8  # Whole column
        assert tensor[17, :, 3].sum() == 0  # Other columns empty


class TestTensorToBoard:
    """Tests for converting tensor back to board."""
    
    def test_round_trip(self):
        """Converting to tensor and back preserves pieces."""
        original = chess.Board()
        tensor = board_to_tensor(original)
        recovered = tensor_to_board(tensor)
        
        # Check all pieces match
        for square in chess.SQUARES:
            original_piece = original.piece_at(square)
            recovered_piece = recovered.piece_at(square)
            
            if original_piece is None:
                assert recovered_piece is None
            else:
                assert recovered_piece is not None
                assert original_piece.piece_type == recovered_piece.piece_type
                assert original_piece.color == recovered_piece.color


class TestFlipBoardTensor:
    """Tests for board flipping."""
    
    def test_flip_swaps_colors(self):
        """Flipping swaps white and black pieces."""
        board = chess.Board()
        tensor = board_to_tensor(board)
        flipped = flip_board_tensor(tensor)
        
        # White pawns become black pawns (flipped vertically)
        # Original white pawns on rank 2 -> flipped black pawns on rank 7
        assert flipped[6, 6, :].sum() == 8  # Was white, now black, flipped
    
    def test_double_flip_identity(self):
        """Flipping twice returns original."""
        board = chess.Board()
        tensor = board_to_tensor(board)
        flipped_twice = flip_board_tensor(flip_board_tensor(tensor))
        
        assert np.allclose(tensor, flipped_twice)


def test_visualize_runs():
    """Visualization functions run without error."""
    board = chess.Board()
    tensor = board_to_tensor(board)
    
    # Should not raise
    visualize_tensor(tensor)
    visualize_tensor(tensor, channel=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])