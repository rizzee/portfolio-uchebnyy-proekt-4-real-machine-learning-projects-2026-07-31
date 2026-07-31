# ML Portfolio Demo

Этот проект представляет собой набор практических примеров машинного обучения на Python. Внутри реализованы базовые алгоритмы для анализа тональности, детекции спама, рекомендательная система и прогнозирование временных рядов. Всё работает локально без использования внешних API.

## Запуск

Сначала установите зависимости:
```bash
pip install -r requirements.txt
```

Запустите основное меню через CLI:
```bash
python main.py
```

## Пример

Пример использования модуля анализа тональности:
```python
from ml_models.sentiment_analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.predict("I love this project!")
print(result)  # Вывод: Positive
```

## Тесты

Для запуска всех тестов используйте:
```bash
python -m unittest discover -s tests -v
```
