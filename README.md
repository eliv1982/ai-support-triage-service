# ai-support-triage-service

Учебный MVP-проект для первичной AI-обработки клиентских обращений. Сервис принимает текст обращения, канал и идентификатор клиента, вызывает LLM для простой классификации и черновика ответа, а затем сохраняет результат в SQLite.

## Что делает проект

- Принимает обращения через `POST /triage`
- Классифицирует обращение по категориям `billing`, `support`, `complaint`, `other`
- Генерирует короткий `draft_reply`
- Возвращает `confidence` и флаг `escalate`
- Сохраняет каждый запрос в таблицу `tickets`
- Использует fallback, если LLM недоступна или вернула некорректный JSON
- Ограничивает частоту запросов по `client_id`

## Архитектура

- `app/main.py` — FastAPI-приложение и endpoints
- `app/schemas.py` — Pydantic-схемы запроса и ответа
- `app/models.py` — SQLAlchemy-модель `tickets`
- `app/database.py` — подключение к SQLite
- `app/services/triage_service.py` — вызов LLM, парсинг, fallback
- `app/rate_limiter.py` — простой in-memory rate limiter
- `app/repository.py` — сохранение тикетов в БД
- `tests/` — pytest-тесты

## Локальный запуск

1. Создайте и активируйте виртуальное окружение:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Установите зависимости:

```bash
pip install -r requirements.txt
```

3. Создайте `.env` на основе примера:

```bash
Copy-Item .env.example .env
```

4. Запустите приложение:

```bash
uvicorn app.main:app --reload
```

После запуска сервис будет доступен на `http://127.0.0.1:8000`, а health-check — на `GET /health`.

## Запуск через Docker

1. Подготовьте `.env`:

```bash
Copy-Item .env.example .env
```

2. Запустите контейнер:

```bash
docker compose up --build
```

## Переменные окружения

- `OPENAI_API_KEY` — ключ для OpenAI-compatible API
- `OPENAI_BASE_URL` — базовый URL провайдера
- `OPENAI_MODEL` — имя модели
- `RATE_LIMIT_PER_MINUTE` — лимит запросов в минуту на один `client_id`
- `DATABASE_URL` — строка подключения к SQLite, по умолчанию `sqlite:///./app.db`

## Пример запроса

```bash
curl.exe -X POST "http://127.0.0.1:8000/triage" -H "Content-Type: application/json" -d "{\"text\":\"I was charged twice for my last invoice.\",\"channel\":\"email\",\"client_id\":\"client-123\"}"
```

Пример ответа:

```json
{
  "category": "billing",
  "draft_reply": "Thank you for contacting us. We are reviewing the duplicate charge and will get back to you shortly.",
  "confidence": "high",
  "escalate": false
}
```

## Как посмотреть SQLite database

Если установлен клиент `sqlite3`, можно открыть базу так:

```bash
sqlite3 app.db
```

Полезные команды внутри:

```sql
.tables
SELECT id, client_id, channel, category, confidence, escalate, error FROM tickets;
```

## Как запустить тесты

```bash
pytest
```

## Демо-сценарий для видео

1. Запустить сервис локально или через Docker
2. Проверить `GET /health`
3. Отправить успешный запрос на `POST /triage`
4. Показать ответ API
5. Открыть `app.db` и показать сохраненную запись в `tickets`
6. Временно сломать или замокать LLM и показать fallback-поведение
7. Отправить несколько запросов подряд с одним `client_id` и показать HTTP `429`

## Примечание

Основной endpoint проекта — `POST /triage`. Если в шаблоне задания встречается `POST /lead`, это техническая опечатка, и ориентироваться нужно именно на `POST /triage`.
