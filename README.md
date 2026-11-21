# Agentic RAG Playground

## 1. Назначение

Локальный сервис на FastAPI для вопросов по документам с учётом ролей пользователя.
Веб-чат поддерживает отдельные диалоги, историю в PostgreSQL, суммаризацию,
потоковые ответы с источниками и режим «Глубокий анализ».

## 2. Быстрый запуск

Нужны Python 3.11+, PostgreSQL с `pg_search` и `pgvector`, API embeddings и языковой
модели. Базовая схема RAG должна быть подготовлена: пользователи, роли, документы,
фрагменты, RLS и функция `public.hybrid_search`. Миграции проекта добавляют чаты,
реестр файлов и память; они не создают базовую схему с нуля.

Из корня проекта, в PowerShell:

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test,ingest]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Заполните `.env` по следующему разделу. В новой копии подготовленной БД примените
`backend/migrations/001_chats.sql` — `005_history_summary.sql` по порядку от
администратора. В текущей рабочей БД они уже применены. Например, для локального Docker:

```powershell
Get-ChildItem migrations/*.sql | Sort-Object Name | ForEach-Object {
    Get-Content -Raw -Encoding utf8 $_.FullName |
        docker exec -i testagenticrag-db-1 psql -v ON_ERROR_STOP=1 -U rag_admin -d rag
    if ($LASTEXITCODE -ne 0) { throw "Ошибка миграции: $($_.Name)" }
}
.\.venv\Scripts\python.exe -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
```

Откройте [веб-чат](http://localhost:8000/), введите `DEV_ADMIN_TOKEN`, выберите
пользователя с доступом к документам и создайте чат. Загрузка документов описана
в [инструкции импорта](backend/INGESTION.md).

## 3. Конфигурация

Все команды backend выполняются из `backend/`, где читается `.env`.
Полный список параметров — [.env.example](backend/.env.example).
Пример подключения к используемой LM Studio:

```dotenv
DATABASE_URL=postgresql://rag_reader:YOUR_PASSWORD@localhost:25432/rag
APP_ENV=local
DEV_AUTH_ENABLED=true
DEV_USER_ID=1
DEV_API_TOKEN=REPLACE_WITH_RANDOM_TOKEN
DEV_ADMIN_TOKEN=REPLACE_WITH_ANOTHER_RANDOM_TOKEN
LLM_PROVIDER=lmstudio
LLM_BASE_URL=http://172.19.0.1:5985/v1
LLM_MODEL=qwen/qwen3.5-9b
EMBEDDING_BASE_URL=http://172.19.0.1:5985/v1
EMBEDDING_MODEL=text-embedding-baai-bge-m3-568m
EMBEDDING_DIMENSIONS=1024
MEMORY_SUMMARY_ENABLED=true
SUMMARY_LLM_PROVIDER=lmstudio
SUMMARY_LLM_BASE_URL=http://172.19.0.1:5985/v1
SUMMARY_LLM_MODEL=qwen/qwen3.5-9b
SUMMARY_LLM_API_KEY=
```

Адреса и ID моделей должны соответствовать вашему серверу. Для каждого токена
сгенерируйте отдельное значение командой
`.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"`.
`DEV_USER_ID` должен существовать. Backend подключается как `rag_reader`, импорт
использует отдельное подключение с правами записи. `.env` с секретами исключён из Git.

Для контекста LM Studio в 4096 токенов начните с `GENERATION_CONTEXT_CHARS=4000`,
`DEEP_CONTEXT_CHARS=4000`, `MEMORY_MAX_CHARS=2000`, `DEEP_MAX_TOKENS=1000`.
Это бюджеты по символам, не точный подсчёт токенов. `LLM_CONTEXT_TOKENS` не меняет
контекст загруженной модели LM Studio. После изменения `.env` перезапустите backend.

`SUMMARY_LLM_*` задают отдельную модель суммаризации, `DEEP_LLM_*` — модель глубокого
режима. При отдельном URL API-ключ задаётся отдельно. Для vLLM используется провайдер
`openai` и URL с `/v1`, для Ollama — `ollama`. Встроенный thinking отключён.

## 4. Архитектура

```text
Веб / API → проверка токена и владельца чата
          → история и сводка → уточнение поискового запроса
          → BGE-M3 → PostgreSQL: RLS + BM25 + векторный поиск
          → LLM → проверка цитат → SSE-ответ и сохранение истории
```

Обычный режим выполняет один поиск. «Глубокий анализ» добавляет планирование,
оценку материалов, проверку ответа и ограниченные повторы. Он допускает частичные
ответы и явно обозначенные предположения по близким материалам.
Система использует управляемый граф со специализированными этапами и настраиваемыми моделями.

API и веб — `backend/app/`, импорт — `backend/app/ingestion/`, CLI — `backend/scripts/`,
миграции — `backend/migrations/`, тесты — `backend/tests/`.
Подробная карта файлов — [ARCHITECTURE.md](backend/ARCHITECTURE.md).

## 5. Использование

В вебе задайте вопрос, например «Какие основные правила описаны в документах?».
Ответ содержит текст и ссылки на доступные фрагменты. Справа показаны источники,
поисковый запрос, сводка истории и этапы глубокого анализа.

В [Swagger](http://localhost:8000/docs) авторизуйтесь через `DEV_API_TOKEN`
и вызовите `POST /ask`:

```json
{"question": "Какие основные правила описаны в документах?", "limit": 5}
```

Ответ содержит `answer` и `sources`. Для истории используйте `DEV_ADMIN_TOKEN`:
создайте чат через `POST /dev/conversations` с `{"user_id": 1}`, затем отправьте
`POST /dev/conversations/{id}/messages` или `/messages/stream`:

```json
{
  "user_id": 1,
  "message": "Какие основные правила описаны в документах?",
  "client_id": "api-test",
  "request_id": "85b9def9-e695-4eb5-b85e-1b184a1f8705",
  "deep_thinking": true,
  "limit": 5
}
```

Для нового сообщения создавайте новый UUID `request_id`. В SSE `delta` — черновик,
`reset` сбрасывает его, `done` содержит итог. Поток с `error` или без `done` не завершён.
`/ask` и `/search` работают без истории чата.

## 6. Проверка и диагностика

Из `backend/`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
Invoke-RestMethod http://localhost:8000/health
```

Обычный запуск тестов использует подмены внешних сервисов. Проверки реальной БД
включаются отдельно через `CHAT_DB_TESTS=1` и `INGEST_DB_TESTS=1` и требуют локального
Docker PostgreSQL и административного доступа для подготовки тестовых данных.
`/health` проверяет процесс, но не готовность моделей и поиска. Логи смотрите
в терминале uvicorn и на сервере моделей, этапы запроса — в панели веб-чата.

| Ошибка | Что проверить |
|---|---|
| `401` | Правильный токен для обычного API или административных `/dev` endpoints |
| `400` от LLM / context limit | ID модели, контекст её загрузки и бюджеты текста |
| `502` / `504` | Доступность моделей, URL, API-ключи, таймауты и лимит ответа |
| `409` / `429` | Занятый чат, повтор `request_id` или лимит параллельных генераций |

## 7. Ограничения и доступ

Сервис предназначен для локальных экспериментов. `DEV_ADMIN_TOKEN` позволяет
действовать от имени разных пользователей; публичная авторизация не реализована.
Документы фильтруются по ролям через RLS, чаты — по владельцу. Связанные ответы
и сводки скрываются после отзыва доступа к исходным документам.

Проверка моделью не гарантирует достоверность; предположения нужно отличать от фактов.
Сводки могут терять детали, старые чаты без сводки обрабатываются с ограниченным
хвостом истории. Разные чаты работают параллельно, в одном чате допускается одна
активная генерация. `MAX_CONCURRENT_GENERATIONS` действует на процесс backend.
