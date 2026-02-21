import math

class CarEnv:
    def __init__(self, track_image, start_x, start_y, start_angle=0):
        self.track_image = track_image
        self.width, self.height = track_image.get_size()
        self.start_x = start_x
        self.start_y = start_y
        self.start_angle = start_angle

        self.car_x = start_x
        self.car_y = start_y
        self.car_angle = start_angle
        self.car_speed = 5 # Auto pojede konstantní rychlostí
        
        # Nastavení senzorů: úhly vůči natočení auta
        self.sensor_angles = [-90, -45, 0, 45, 90]
        self.max_sensor_distance = 200 # Maximální dohled (v pixelech)
        self.sensor_points = [] # Pro vizualizaci v main.py

    def get_sensors(self):
        distances = []
        self.sensor_points = []

        for angle_offset in self.sensor_angles:
            # Vypočítáme absolutní úhel senzoru
            angle = self.car_angle + angle_offset
            rad = math.radians(angle)
            
            dist = 0
            hit_x, hit_y = self.car_x, self.car_y
            
            # Postupujeme bod po bodu podél paprsku
            for d in range(1, self.max_sensor_distance + 1):
                check_x = int(self.car_x + d * math.cos(rad))
                check_y = int(self.car_y - d * math.sin(rad)) # Mínus, protože osa Y roste dolů
                
                # Ochrana proti vyjetí z obrazovky (IndexError)
                if check_x < 0 or check_x >= self.width or check_y < 0 or check_y >= self.height:
                    dist = d
                    hit_x, hit_y = check_x, check_y
                    break
                    
                pixel_color = self.track_image.get_at((check_x, check_y))
                # Pokud pixel není bílý (trať), narazili jsme na mantinel
                if pixel_color[:3] != (255, 255, 255):
                    dist = d
                    hit_x, hit_y = check_x, check_y
                    break
            else:
                # Pokud paprsek na nic nenarazil, nastavíme maximum
                dist = self.max_sensor_distance
                hit_x = self.car_x + dist * math.cos(rad)
                hit_y = self.car_y - dist * math.sin(rad)
                
            # Normalizace vzdálenosti (hodnota 0.0 až 1.0) pro neuronovou síť
            distances.append(dist / self.max_sensor_distance)
            self.sensor_points.append((hit_x, hit_y)) # Uložíme si souřadnice dotyku pro kreslení
            
        return distances

    def reset(self):
        self.car_x = self.start_x
        self.car_y = self.start_y
        self.car_angle = self.start_angle
        
        sensors = self.get_sensors()
        # Stav pro síť: 5x hodnota ze senzorů
        state = sensors 
        return state

    def step(self, action):
        # Přemapování akcí pro snazší učení (auto jede pořád dopředu)
        # 0 = Jede rovně
        # 1 = Zatáčí doleva
        # 2 = Zatáčí doprava
        if action == 1:  
            self.car_angle += 5
        elif action == 2:  
            self.car_angle -= 5

        rad = math.radians(self.car_angle)
        self.car_x += self.car_speed * math.cos(rad)
        self.car_y -= self.car_speed * math.sin(rad)

        done = False
        reward = 0.1 # Odměna za přežití

        # Kontrola kolize samotného auta
        if self.car_x < 0 or self.car_x >= self.width or self.car_y < 0 or self.car_y >= self.height:
            done = True
            reward = -10
        else:
            pixel_color = self.track_image.get_at((int(self.car_x), int(self.car_y)))
            if pixel_color[:3] != (255, 255, 255):
                done = True
                reward = -10

        sensors = self.get_sensors()
        state = sensors # Nový stav po provedení akce
        
        return state, reward, done