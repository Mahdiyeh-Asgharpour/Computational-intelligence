import numpy as np
import random
# تنظیمات اولیه
num_locations = 10  # تعداد مکان‌ها
num_vehicles = 3    # تعداد خودروها
vehicle_capacity = 20  # ظرفیت خودروها
# تولید مختصات مکان‌ها
locations = np.random.rand(num_locations, 2) * 100  # مختصات مکان‌ها (x, y)
priorities = np.random.choice([0, 1], num_locations)  # 1 برای مکان‌های با اولویت بالا
demands = np.random.randint(1, 10, num_locations)  # تقاضای بسته‌ها
traffic_factors = np.random.rand(num_locations, num_locations) * 2  # ضرایب ترافیکی
# محاسبه ماتریس فاصله
def calculate_distance_matrix(locations):
    num_locations = len(locations)
    distance_matrix = np.zeros((num_locations, num_locations))
    for i in range(num_locations):
        for j in range(num_locations):
            distance_matrix[i][j] = np.sqrt((locations[i][0] - locations[j][0])**2 +
                                            (locations[i][1] - locations[j][1])**2)
    return distance_matrix
distance_matrix = calculate_distance_matrix(locations)
# تابع برازش
def fitness(route, distance_matrix, priorities, demands, vehicle_capacity):
    total_distance = 0
    penalty_delays = 0
    penalty_overcapacity = 0
    penalty_time_limit = 0
    current_capacity = 0
    time_elapsed = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i]][route[i + 1]]
        time_elapsed += traffic_factors[route[i]][route[i + 1]] * distance_matrix[route[i]][route[i + 1]]
        # جریمه برای تأخیر در تحویل بسته‌های اولویت بالا
        if priorities[route[i]] == 1 and time_elapsed > 120:
            penalty_delays += 10
        # بررسی ظرفیت خودرو
        current_capacity += demands[route[i]]
        if current_capacity > vehicle_capacity:
            penalty_overcapacity += 10
    # جریمه برای زمان کل مسیر بیشتر از 8 ساعت
    if time_elapsed > 480:  # 8 ساعت = 480 دقیقه
        penalty_time_limit += 100
    fitness_value = total_distance + penalty_delays + penalty_overcapacity + penalty_time_limit
    return fitness_value
# عملگرهای ژنتیک
def crossover(parent1, parent2):
    split_point = len(parent1) // 2
    child = parent1[:split_point] + [gene for gene in parent2 if gene not in parent1[:split_point]]
    return child
def mutate(route):
    i, j = random.sample(range(len(route)), 2)
    route[i], route[j] = route[j], route[i]
# تابع برای به‌روزرسانی ضرایب ترافیکی
def update_traffic_factors():
    global traffic_factors
    traffic_factors = np.random.rand(num_locations, num_locations) * 2
    print("Traffic factors updated.")
# الگوریتم ژنتیک
def genetic_algorithm(distance_matrix, priorities, demands, vehicle_capacity, population_size=50, generations=100):
    population = [random.sample(range(num_locations), num_locations) for _ in range(population_size)]
    for generation in range(generations):
        population = sorted(population, key=lambda route: fitness(route, distance_matrix, priorities, demands, vehicle_capacity))
        next_population = population[:10]  # انتخاب برترین‌ها
        while len(next_population) < population_size:
            parent1, parent2 = random.sample(population[:20], 2)
            child = crossover(parent1, parent2)
            if random.random() < 0.1:  # احتمال جهش
                mutate(child)
            next_population.append(child)
        population = next_population
        best_fitness = fitness(population[0], distance_matrix, priorities, demands, vehicle_capacity)
        print(f"Generation {generation}, Best Fitness: {best_fitness}")
        # به‌روزرسانی ضرایب ترافیکی در هر 10 نسل
        if generation % 10 == 0:
            update_traffic_factors()
    return population[0]
# اجرا
best_route = genetic_algorithm(distance_matrix, priorities, demands, vehicle_capacity)
print("Best Route:", best_route)
