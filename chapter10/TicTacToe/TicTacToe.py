class TicTacToe:
    def __init__(self):
        self.board = list('_' * 9)
        self.result = 0
        self.player = 1
        self.winning_cases = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

    def display_board(self):
        print('Game Board')
        boardDisplay = [f' {self.board[i]} ' if self.board[i] != '_' else f'({str(i)})' for i in range(9)]
        for i in range(3):
            print(' '.join(boardDisplay[3 * i:3 * (i + 1)]))

    def player_input(self):
        input_msg = 'Please select your next position:'
        v = int(input(input_msg))
        if self.board[v] != '_':
            print("That space can been taken")
        self.board[v] = 'X'

    def ai_input(self, v):
        self.board[v] = "O"
        print(f'AI chose {v}')

    def check_result(self):
        for w in self.winning_cases:
            if self.board[w[0]] != '_' and self.board[w[0]] == self.board[w[1]] and self.board[w[1]] == self.board[w[2]]:
                if self.board[w[0]] == 'X':
                    self.result = 1
                else:
                    self.result = 2
        if '_' not in self.board:
            self.result = 3

    def switch_player(self, v):
        self.player = 3 - self.player
        if self.player == 2:
            self.ai_input(v)
        else:
            self.player_input()
