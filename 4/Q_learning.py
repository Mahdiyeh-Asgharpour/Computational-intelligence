import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, accuracy_score
genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi']
num_items = 20
items = []
for item_id in range(num_items):
    item = {
        'item_id': item_id,
        'title': f"Item {item_id}",
        'genre': np.random.choice(genres),
        'popularity': np.random.randint(1, 100),
        'rating': round(np.random.uniform(1, 5), 1),
        'length': np.random.randint(60, 180)
    }
    items.append(item)
num_users = 100 
user_interactions = {user_id: [] for user_id in range(num_users)}
def add_interaction(user_id, item_id, feedback):
    interaction = {'item_id': item_id, 'feedback': feedback}
    user_interactions[user_id].append(interaction)
for user_id in range(num_users):
    for _ in range(np.random.randint(5, 10)):
        item_id = np.random.randint(0, num_items)
        feedback = np.random.choice(['like', 'dislike'])
        add_interaction(user_id, item_id, feedback)
states = [(user_id, item['item_id']) for user_id in range(num_users) for item in items]
actions = [item['item_id'] for item in items]
Q_table = np.zeros((len(states), len(actions)))
alpha = 0.1
gamma = 0.9
epsilon = 0.2
def reward(user_id, item_id):
    for interaction in user_interactions[user_id]:
        if interaction['item_id'] == item_id:
            return 10 if interaction['feedback'] == 'like' else -10
    return -1
def train_q_learning(episodes=1000):
    global Q_table
    for episode in range(episodes):
        user_id = np.random.randint(0, num_users)
        current_state = (user_id, np.random.choice(actions))
        state_index = states.index(current_state)
        for _ in range(10):
            if random.uniform(0, 1) < epsilon:
                action = np.random.choice(actions)
            else:
                action = actions[np.argmax(Q_table[state_index])]
            r = reward(current_state[0], action)
            action_index = actions.index(action)
            new_state = (current_state[0], action)
            new_state_index = states.index(new_state)
            Q_table[state_index, action_index] += alpha * (
                r + gamma * np.max(Q_table[new_state_index]) - Q_table[state_index, action_index]
            )
            current_state = new_state
            state_index = new_state_index
def recommend(user_id):
    user_states = [(user_id, item_id) for item_id in actions]
    user_state_indices = [states.index(state) for state in user_states]
    best_action = actions[np.argmax([Q_table[idx].max() for idx in user_state_indices])]
    recommended_item = next(item for item in items if item['item_id'] == best_action)
    return recommended_item
test_user_interactions = {user_id: [] for user_id in range(num_users)}
for user_id in range(num_users):
    for _ in range(np.random.randint(5, 10)):
        item_id = np.random.randint(0, num_items)
        feedback = np.random.choice(['like', 'dislike'])
        test_user_interactions[user_id].append({'item_id': item_id, 'feedback': feedback})
def evaluate_recommendations():
    y_true = []
    y_pred = []
    for user_id, interactions in test_user_interactions.items():
        recommended_item = recommend(user_id)
        recommended_item_id = recommended_item['item_id']
        for interaction in interactions:
            y_true.append(1 if interaction['feedback'] == 'like' else 0)
            y_pred.append(1 if interaction['item_id'] == recommended_item_id else 0)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
def improve_system():
    global alpha, gamma, epsilon
    alpha = 0.05
    gamma = 0.95
    epsilon = max(epsilon - 0.01, 0.1)
    train_q_learning(episodes=1000)
    print("System retrained with improved parameters.")
print("Training the system...")
train_q_learning(episodes=1000)
print("\nRecommendations for All Users:")
for user_id in range(num_users):
    recommended = recommend(user_id)
    print(f"User {user_id}: {recommended}")
print("\nInitial Evaluation:")
evaluate_recommendations()
print("\nImproving the System...")
improve_system()
print("\nEvaluation After Improvement:")
evaluate_recommendations()
