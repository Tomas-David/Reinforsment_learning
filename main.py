import pygame
import sys
from CarEvn import CarEnv
from agent import DQNAgent

pygame.init()
pygame.font.init() 

WIDTH, HEIGHT = 2000, 1000
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("F1 Tratě - Trénink AI")
clock = pygame.time.Clock()

font = pygame.font.SysFont("Arial", 26, bold=True)

track_image = pygame.image.load("./assets/track.png").convert()
track_image = pygame.transform.scale(track_image, (WIDTH, HEIGHT))

car_image = pygame.image.load("./assets/f1.png").convert_alpha()
car_image = pygame.transform.scale(car_image, (60, 30))

env = CarEnv(track_image, start_x=300, start_y=890, start_angle=0)
agent = DQNAgent()

state = env.reset()
score = 0
record = 0
episode = 1


total_start_time = pygame.time.get_ticks() 
episode_start_time = pygame.time.get_ticks() 

running = True
FPS = 60 

WINNING_SCORE = 2000 
track_completed = False

while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state_old = state
    action = agent.get_action(state_old)
    state_new, reward, done = env.step(action)
    score += reward
    
    agent.remember(state_old, action, reward, state_new, done)
    agent.train_experience_replay(batch_size=64)
    state = state_new
    
    current_time = pygame.time.get_ticks()
    episode_time_sec = (current_time - episode_start_time) / 1000
    total_time_sec = (current_time - total_start_time) / 1000

    if score >= WINNING_SCORE and not track_completed:
        track_completed = True
        print(f"ÚSPĚCH! Trať projeta bezchybně po {total_time_sec:.1f} sekundách celkového tréninku (Pokus č. {episode})!")

    if done:
        state = env.reset()
        if score > record:
            record = score
            
        print(f"Pokus: {episode} | Skóre: {score:.1f} | Čas jízdy: {episode_time_sec:.1f}s | Rekord: {record:.1f} | Epsilon: {agent.epsilon:.3f}")
        
        score = 0
        episode += 1
        episode_start_time = pygame.time.get_ticks() # Reset času pro nový pokus

    screen.blit(track_image, (0, 0))
    
    for point in env.sensor_points:
        pygame.draw.line(screen, (0, 255, 0), (int(env.car_x), int(env.car_y)), (int(point[0]), int(point[1])), 2)
        pygame.draw.circle(screen, (255, 0, 0), (int(point[0]), int(point[1])), 4)

    rotated_car = pygame.transform.rotate(car_image, env.car_angle)
    rect = rotated_car.get_rect(center=(int(env.car_x), int(env.car_y)))
    screen.blit(rotated_car, rect)


    hud_texts = [
        f"Pokus (Epizoda): {episode}",
        f"Skóre aktuální: {score:.1f}",
        f"Rekordní skóre: {record:.1f}",
        f"Čas aktuální jízdy: {episode_time_sec:.1f} s",
        f"Celkový čas tréninku: {total_time_sec:.1f} s",
        f"Náhodnost (Epsilon): {agent.epsilon:.5f}"
    ]

    for i, text_str in enumerate(hud_texts):
        text_shadow = font.render(text_str, True, (0, 0, 0))
        screen.blit(text_shadow, (22, 22 + i * 35))
        text_surface = font.render(text_str, True, (255, 255, 0))
        screen.blit(text_surface, (20, 20 + i * 35))

    if track_completed:
        win_text = font.render("TRAŤ PROJETA BEZCHYBNĚ!", True, (0, 255, 0))
        screen.blit(win_text, (WIDTH//2 - 150, 50))

    pygame.display.flip()

pygame.quit()
sys.exit() 