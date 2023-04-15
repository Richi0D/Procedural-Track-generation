# Import and initialize the pygame library
import pygame
import os
import importlib

gen = importlib.import_module('track-generation-sat')

# Set the width and height of the output window, in pixels
WIDTH = 800
HEIGHT = 800
STARTPOSITION = (WIDTH / 2, HEIGHT - 600)
LENGTH = 15
PIECESIZE = 50


class Piece(pygame.Surface):
    """
    create and draw different pieces
    we have: start = b, finish = g, straight = s, left = l, right = r
    """

    def __init__(self, typ: str, size: int = 50):
        self.size = size
        self.imagepath = os.fspath('images')

        # Load the image, preserve alpha channel for transparency
        if typ == 'b':
            self.surf = pygame.image.load(os.path.join(self.imagepath, 'start-simple.png')).convert_alpha()
        elif typ == 'g':
            self.surf = pygame.image.load(os.path.join(self.imagepath, 'finish-simple.png')).convert_alpha()
        elif typ == 's':
            self.surf = pygame.image.load(os.path.join(self.imagepath, 'straight-simple.png')).convert_alpha()
        elif typ == 'l':
            self.surf = pygame.image.load(os.path.join(self.imagepath, 'left-simple.png')).convert_alpha()
        elif typ == 'r':
            self.surf = pygame.image.load(os.path.join(self.imagepath, 'right-simple.png')).convert_alpha()

        # scale image to correct size
        self.surf = pygame.transform.scale(self.surf, (self.size, self.size))
        # Save the rect, so you can move it
        self.rect = self.surf.get_rect()

    def update(self, pos: tuple):
        self.rect.center = pos

    def rotate(self, angle):
        rotated = pygame.transform.rotate(self.surf, angle)
        self.rect = rotated.get_rect(center=self.surf.get_rect(center=self.rect.center).center)
        self.surf = rotated


class Button:
    """
    Create a button, then blit the surface in the while loop
    """

    def __init__(self, text, pos, font, textcolor='white', bg="black", feedback=""):
        self.x, self.y = pos
        self.font = pygame.font.SysFont("Arial", font)
        if feedback == "":
            self.feedback = "text"
        else:
            self.feedback = feedback
        self.change_text(text, textcolor, bg)

    def change_text(self, text, textcolor='white', bg="black"):
        """Change the text when you click"""
        self.text = self.font.render(text, 1, pygame.Color(textcolor))
        self.size = (self.text.get_size()[0] + 10, self.text.get_size()[1])
        self.surface = pygame.Surface(self.size)
        self.surface.fill(bg)
        self.surface.blit(self.text, (5, 0))
        self.rect = pygame.Rect(self.x, self.y, self.size[0], self.size[1])

    def show(self):
        screen.blit(self.surface, (self.x, self.y))


def draw_track(piece_list: list, start: tuple):
    """draw every piece"""
    angles = [0, 90, 180, 270]
    current_angle = 0  # list index for angles
    current_position = start
    col = []
    for p in piece_list:
        typ = p[0]  # extract type of piece

        new_piece = Piece(typ)
        new_piece.update(current_position)
        new_piece.rotate(angles[-current_angle])
        # screen.blit(new_piece.surf, new_piece.rect)
        col.append((new_piece.surf, new_piece.rect))

        # calculate next position of piece based on angle
        # we only need to update angle if we go left or right
        if typ == 'l':
            if current_angle == 0:
                current_angle = 3
            else:
                current_angle -= 1
        if typ == 'r':
            if current_angle == 3:
                current_angle = 0
            else:
                current_angle += 1

        if angles[current_angle] == 0:
            # go up
            current_position = (current_position[0], current_position[1] - PIECESIZE)
        if angles[current_angle] == 180:
            # go down
            current_position = (current_position[0], current_position[1] + PIECESIZE)
        if angles[current_angle] == 90:
            # go left
            current_position = (current_position[0] + PIECESIZE, current_position[1])
        if angles[current_angle] == 270:
            # go right
            current_position = (current_position[0] - PIECESIZE, current_position[1])
    return col


if __name__ == "__main__":
    # Initialize the Pygame engine
    pygame.init()
    # Set up the drawing window and clock
    screen = pygame.display.set_mode(size=[WIDTH, HEIGHT])
    clock = pygame.time.Clock()

    # Generate problem
    generator = gen.Generator(length=LENGTH)
    generator.generate_problem()  # generate problem

    # list of track pieces
    track = []

    # create buttons
    gen_button = Button("Generate",
                        (10, 10),
                        font=25,
                        bg="navy",
                        textcolor='white',
                        feedback="")
    # Create text. Is basically the same as a button just we do not check if we press it
    start_text = Button(f'Startposition: {STARTPOSITION}',
                        (120, 10),
                        font=20,
                        textcolor='black',
                        bg='grey')
    length_text = Button(f'Track length: {LENGTH}',
                        (400, 10),
                        font=20,
                        textcolor='black',
                        bg='grey')

    # Run until you get to an end condition
    running = True
    while running:
        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # check if quit
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_PLUS:
                    LENGTH += 1
                    length_text.change_text(f'Track length: {LENGTH}', textcolor='black', bg='grey')
                    generator = gen.Generator(length=LENGTH)
                    generator.generate_problem()  # generate problem
                if event.key == pygame.K_MINUS:
                    LENGTH -= 1
                    length_text.change_text(f'Track length: {LENGTH}', textcolor='black', bg='grey')
                    generator = gen.Generator(length=LENGTH)
                    generator.generate_problem()  # generate problem
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if pygame.mouse.get_pressed()[0]:
                    # checks for left click of mouse
                    if gen_button.rect.collidepoint(mouse_x, mouse_y):
                        pieces = generator.get_true_literals()  # get solution to our problem
                        track = draw_track(pieces, STARTPOSITION)
                    else:
                        start_text.change_text(f'Startposition: {STARTPOSITION}', textcolor='black', bg='grey')
                        STARTPOSITION = (mouse_x, mouse_y)

        # To render the screen, first fill the background with pink
        screen.fill(pygame.Color('white'))

        # show buttons
        gen_button.show()
        # show text
        start_text.show()
        length_text.show()

        # show track
        for t in track:
            screen.blit(t[0], t[1])

        # Flip the display to make everything appear
        pygame.display.flip()
        clock.tick(60)

    # Done! Time to quit.
    pygame.quit()
