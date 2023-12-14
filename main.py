import random


# Генерация n случайных точек с линейной зависимостью
def generate_data(n):
    data = []
    for _ in range(n):
        x = random.uniform(0, 10)
        y = 2 * x + 1 + random.gauss(0, 1)  # Линейная зависимость с шумом
        data.append((x, y))
    return data


# Реализация линейной регрессии с градиентным спуском
def linear_regression(data, learning_rate, epochs):
    # Инициализация случайных коэффициентов
    theta0, theta1 = random.uniform(0, 1), random.uniform(0, 1)

    # Функция предсказания
    predict = lambda x: theta0 + theta1 * x

    # Функция стоимости (MSE)
    cost_function = lambda x, y: (predict(x) - y) ** 2

    # Градиенты по коэффициентам
    gradient_theta0 = lambda x, y: 2 * (predict(x) - y)
    gradient_theta1 = lambda x, y: 2 * (predict(x) - y) * x

    # Обновление параметров с использованием градиентного спуска
    update_parameters = lambda x, y, learning_rate: (
        theta0 - learning_rate * gradient_theta0(x, y),
        theta1 - learning_rate * gradient_theta1(x, y)
    )

    # Обучение модели
    for epoch in range(epochs):
        for x, y in data:
            theta0, theta1 = update_parameters(x, y, learning_rate)

    # Возвращаем обученные коэффициенты
    return theta0, theta1


def main():
    data = generate_data(100)
    learning_rate = 0.01
    epochs = 1000
    theta0, theta1 = linear_regression(data, learning_rate, epochs)

    print(f"Обученные коэффициенты: theta0 = {theta0}, theta1 = {theta1}")
    predict = lambda x: theta0 + theta1 * x
    print(predict(10))


if __name__ == '__main__':
    main()