import pygame
import numpy as np
from generate import Generator

gen = Generator()

pygame.init()

window_size = (1280, 720)
screen = pygame.display.set_mode(window_size)

clock = pygame.time.Clock()

font = pygame.font.SysFont("Arial", 24)

image = gen.generate(0, 0)
image_scaled = pygame.surfarray.make_surface(np.kron(image, np.ones((40, 40))))

while True:
    mouse_pos = pygame.mouse.get_pos()
    screen.fill((255, 255, 255))

    x = (mouse_pos[0] - window_size[0] / 2) / window_size[0] * 6 
    y = (mouse_pos[1] - window_size[1] / 2) / window_size[1] * 6


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()


    image = gen.generate(x, y)
    image = np.transpose(image, (1, 0))
    image_scaled = pygame.surfarray.make_surface(np.kron(image, np.ones((20, 20))))
    #image_scaled = pygame.surfarray.make_surface(np.zeros((100, 100)) * 255)
    
    screen.blit(image_scaled, (window_size[0] / 2 - 14 * 20, 50))

    
    fps = clock.get_fps()
    text = font.render(f"x: {x:.3f} y: {y:.3f} fps: {fps:.2f}", True, (0, 0, 0))
    screen.blit(text, (10, 10))

    pygame.display.update()

    clock.tick(120)

pygame.quit()

