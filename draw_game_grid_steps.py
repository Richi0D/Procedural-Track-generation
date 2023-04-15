# Import and initialize the pygame library
import pygame
import os
import importlib
import numpy as np

gen = importlib.import_module('track-generation-grid-steps')

# Set the width and height of the output window, in pixels
WIDTH = 800
HEIGHT = 900

COLOR_INACTIVE = pygame.Color('lightskyblue3')  # color for textbox
COLOR_ACTIVE = pygame.Color('dodgerblue2')  # color for textbox
pygame.font.init()
FONT = pygame.font.SysFont("Arial", 20)


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


class InputBox:
    def __init__(self, x, y, w, h, text=''):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = COLOR_INACTIVE
        self.text = text
        self.txt_surface = FONT.render(text, True, self.color)
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # If the user clicked on the input_box rect.
            if self.rect.collidepoint(event.pos):
                # Toggle the active variable.
                self.active = not self.active
            else:
                self.active = False
            # Change the current color of the input box.
            self.color = COLOR_ACTIVE if self.active else COLOR_INACTIVE
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    print(self.text)
                    self.text = ''
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                # Re-render the text.
                self.txt_surface = FONT.render(self.text, True, self.color)

    def update(self):
        # Resize the box if the text is too long.
        width = min(200, self.txt_surface.get_width() + 10)
        self.rect.w = width

    def show(self, screen):
        # Blit the text.
        screen.blit(self.txt_surface, (self.rect.x + 5, self.rect.y + 5))
        # Blit the rect.
        pygame.draw.rect(screen, self.color, self.rect, 2)


def define_typ_rotation(step, pos, pos_next, pos_old):
    typ = 's'
    rotation = 0  # standard value
    if step == 1:
        typ = 'b'
        # find direction of current piece
        deg = np.arctan2(pos_next[0] - pos[0], pos_next[1] - pos[1]) * (180 / np.pi)
        rotation = deg + 90
    elif step == tracklength:
        typ = 'g'
        deg = np.arctan2(pos[0] - pos_old[0], pos[1] - pos_old[1]) * (180 / np.pi)
        rotation = deg + 90
    else:
        deg_long = np.arctan2(pos_next[0] - pos_old[0], pos_next[1] - pos_old[1]) * (180 / np.pi)
        deg_short = np.arctan2(pos_next[0] - pos[0], pos_next[1] - pos[1]) * (180 / np.pi)
        if deg_long == 0 or deg_long == 180:
            typ = 's'
            rotation = 90
        elif deg_long == 90 or deg_long == -90:
            typ = 's'
            rotation = 0
        # 8 possibilities
        elif (deg_long == -135 and deg_short == 180) or (deg_long == 45 and deg_short == 90):
            typ = 'l'
            rotation = 0
        elif (deg_long == 135 and deg_short == 180) or (deg_long == -45 and deg_short == -90):
            typ = 'l'
            rotation = 90
        elif (deg_long == -135 and deg_short == -90) or (deg_long == 45 and deg_short == 0):
            typ = 'l'
            rotation = 180
        elif (deg_long == 135 and deg_short == 90) or (deg_long == -45 and deg_short == 0):
            typ = 'l'
            rotation = 270
    return typ, rotation


def draw_track(sol, grid, piecesize):
    """draw every piece"""
    col = []
    pos_old = None
    for i in range(1, tracklength + 1):
        pos = np.where(sol == i)
        pos = (pos[0][0], pos[1][0])
        if i != tracklength:
            pos_next = np.where(sol == i + 1)
            pos_next = (pos_next[0][0], pos_next[1][0])
        typ, rotation = define_typ_rotation(i, pos, pos_next, pos_old)
        new_piece = Piece(typ, piecesize)
        new_piece.update(((pos[1] - 1) * piecesize + (piecesize / 2), (pos[0] - 1) * piecesize + 50 + (piecesize / 2)))
        new_piece.rotate(-rotation)
        col.append((new_piece.surf, new_piece.rect))
        pos_old = pos
    return col


if __name__ == "__main__":
    # Initialize the Pygame engine
    pygame.init()
    # Set up the drawing window and clock
    screen = pygame.display.set_mode(size=[WIDTH, HEIGHT])
    clock = pygame.time.Clock()

    # init Generator
    gridsize = [5, 5]
    tracklength = 25
    piecesize = int(WIDTH / max(gridsize))
    generator = gen.Generator_grid(grid_size=gridsize, length=tracklength)

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
    plus_button_grid = Button(f'+',
                              (120, 10),
                              font=25,
                              textcolor='white',
                              bg='navy')
    minus_button_grid = Button(f' - ',
                               (150, 10),
                               font=25,  # ä
                               textcolor='white',
                               bg='navy')
    grid_text = Button(f'Grid size: {gridsize[0]}x{gridsize[1]}',
                       (200, 10),
                       font=20,
                       textcolor='black',
                       bg='grey')
    plus_button_length = Button(f'+',
                                (400, 10),
                                font=25,
                                textcolor='white',
                                bg='navy')
    minus_button_length = Button(f' - ',
                                 (430, 10),
                                 font=25,  # ä
                                 textcolor='white',
                                 bg='navy')
    length_text = Button(f'Track length: {tracklength}',
                         (480, 10),
                         font=20,
                         textcolor='black',
                         bg='grey')

    buttons = [gen_button, plus_button_grid, minus_button_grid, grid_text, plus_button_length, minus_button_length,
               length_text]
    # Create input Boxes

    # Run until you get to an end condition
    running = True
    while running:
        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # check if quit
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if pygame.mouse.get_pressed()[0]:
                    # checks for left click of mouse
                    if gen_button.rect.collidepoint(mouse_x, mouse_y):
                        if len(generator.problem) == 0 or (
                                generator.grid_size[0] - 2 != gridsize[0] and generator.grid_size[1] - 2 != gridsize[
                            1]) or generator.length != tracklength:
                            generator.generate_problem()  # generate problem
                        sol, grid = generator.get_solution()
                        track = draw_track(sol, grid, piecesize)

                    if plus_button_grid.rect.collidepoint(mouse_x, mouse_y):
                        gridsize[0] += 1
                        gridsize[1] += 1
                        grid_text.change_text(f'Grid size: {gridsize[0]}x{gridsize[1]}', textcolor='black', bg='grey')
                        piecesize = int(WIDTH / max(gridsize))
                        generator = gen.Generator_grid(grid_size=gridsize, length=tracklength)
                    if minus_button_grid.rect.collidepoint(mouse_x, mouse_y):
                        gridsize[0] -= 1
                        gridsize[1] -= 1
                        grid_text.change_text(f'Grid size: {gridsize[0]}x{gridsize[1]}', textcolor='black', bg='grey')
                        piecesize = int(WIDTH / max(gridsize))
                        generator = gen.Generator_grid(grid_size=gridsize, length=tracklength)

                    if plus_button_length.rect.collidepoint(mouse_x, mouse_y):
                        tracklength += 1
                        length_text.change_text(f'Track length: {tracklength}', textcolor='black', bg='grey')
                        generator = gen.Generator_grid(grid_size=gridsize, length=tracklength)
                    if minus_button_length.rect.collidepoint(mouse_x, mouse_y):
                        tracklength -= 1
                        length_text.change_text(f'Track length: {tracklength}', textcolor='black', bg='grey')
                        generator = gen.Generator_grid(grid_size=gridsize, length=tracklength)

        # To render the screen, first fill the background with pink
        screen.fill(pygame.Color('grey'))

        # show buttons and text
        for b in buttons:
            b.show()

        # show track
        for t in track:
            screen.blit(t[0], t[1])

        # Flip the display to make everything appear
        pygame.display.flip()
        clock.tick(60)

    # Done! Time to quit.
    pygame.quit()
