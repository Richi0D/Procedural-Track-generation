# Import and initialize the pygame library
import pygame
import os
import importlib
import numpy as np

gen = importlib.import_module('track-generation-grid-all')

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
        width = min(200, self.txt_surface.get_width()+10)
        self.rect.w = width

    def show(self, screen):
        # Blit the text.
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
        # Blit the rect.
        pygame.draw.rect(screen, self.color, self.rect, 2)

def define_typ_rotation(arr):
    typ = 's'
    rotation = 0
    trues = np.nonzero(arr[1:])[0]
    if len(trues) > 1:
        if trues[0] == 0 and trues[1] == 1:
            typ = 's'
            rotation = 0
        elif trues[0] == 2 and trues[1] == 3:
            typ = 's'
            rotation = 90
        elif trues[0] == 1 and trues[1] == 2:
            typ = 'l'
            rotation = 0
        elif trues[0] == 0 and trues[1] == 2:
            typ = 'l'
            rotation = 90
        elif trues[0] == 0 and trues[1] == 3:
            typ = 'l'
            rotation = 180
        elif trues[0] == 1 and trues[1] == 3:
            typ = 'l'
            rotation = 270
    else:
        typ = 'bg'
        if trues[0] == 0:
            rotation = 0
        elif trues[0] == 1:
            rotation = 180
        elif trues[0] == 2:
            rotation = 270
        elif trues[0] == 3:
            rotation = 90
    return typ, rotation

def draw_track(sol, grid, piecesize, start, end):
    """draw every piece"""
    # index: 's'=0, 'u'=1, 'd'=2, 'l'=3, 'r'=4
    col = []
    for ij in np.ndindex(sol.shape[:2]):
        if grid[ij] == 1:
            typ, rotation = define_typ_rotation(sol[ij])
            if ij == start:
                typ = 'b'
            elif ij == end:
                rotation += 180
                typ = 'g'

            new_piece = Piece(typ, piecesize)
            new_piece.update(((ij[1]-1)*piecesize+(piecesize/2), (ij[0]-1)*piecesize+50+(piecesize/2)))
            new_piece.rotate(-rotation)
            col.append((new_piece.surf, new_piece.rect))
    return col


if __name__ == "__main__":
    # Initialize the Pygame engine
    pygame.init()
    # Set up the drawing window and clock
    screen = pygame.display.set_mode(size=[WIDTH, HEIGHT])
    clock = pygame.time.Clock()

    # init Generator
    gridsize = [10, 10]
    piecesize = int(WIDTH / max(gridsize))
    generator = gen.Generator_grid(grid_size=gridsize)

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
    plus_button = Button(f'+',
                        (120, 10),
                        font=25,
                        textcolor='white',
                        bg='navy')
    minus_button = Button(f' - ',
                        (150, 10),
                        font=25,
                        textcolor='white',
                        bg='navy')
    grid_text = Button(f'Grid size: {gridsize[0]}x{gridsize[1]}',
                        (200, 10),
                        font=20,
                        textcolor='black',
                        bg='grey')
    buttons = [gen_button, plus_button, minus_button, grid_text]
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
                        if len(generator.problem) == 0 or (generator.grid_size[0] - 2 != gridsize[0] and generator.grid_size[1] - 2 != gridsize[1]):
                            generator.generate_problem()  # generate problem
                        sol, grid = generator.get_solution()
                        pos_start = generator.start
                        pos_end = generator.end
                        track = draw_track(sol, grid, piecesize, pos_start, pos_end)
                    if plus_button.rect.collidepoint(mouse_x, mouse_y):
                        gridsize[0] += 1
                        gridsize[1] += 1
                        grid_text.change_text(f'Grid size: {gridsize[0]}x{gridsize[1]}', textcolor='black', bg='grey')
                        piecesize = int(WIDTH / max(gridsize))
                        generator = gen.Generator_grid(grid_size=gridsize)
                    if minus_button.rect.collidepoint(mouse_x, mouse_y):
                        gridsize[0] -= 1
                        gridsize[1] -= 1
                        grid_text.change_text(f'Grid size: {gridsize[0]}x{gridsize[1]}', textcolor='black', bg='grey')
                        piecesize = int(WIDTH / max(gridsize))
                        generator = gen.Generator_grid(grid_size=gridsize)

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
