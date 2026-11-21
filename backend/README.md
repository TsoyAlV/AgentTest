# Локальный RAG backend

## Суммаризация истории и отключённый thinking

В текущем `.env` включена накопительная суммаризация перед новым сообщением:

```dotenv
MEMORY_SUMMARY_ENABLED=true
SUMMARY_LLM_PROVIDER=lmstudio
SUMMARY_LLM_BASE_URL=http://172.19.0.1:5985/v1
SUMMARY_LLM_MODEL=qwen/qwen3.5-9b
SUMMARY_LLM_API_KEY=
SUMMARY_INPUT_CHARS=4000
SUMMARY_MAX_CHARS=1000
SUMMARY_MAX_TOKENS=384
```

`SUMMARY_LLM_API_KEY` — ключ API сервера суммаризации, если на нём включена авторизация.
Сейчас он пустой. Модель и сервер можно менять независимо от основной LLM.
Если отдельный URL указан, основной ключ автоматически туда не передаётся.
При пустых настройках модели/провайдера/URL используются основные настройки.

Перед каждым сообщением суммаризатор сжимает предыдущую сводку и новые завершённые
реплики. Основной модели и преобразованию поискового запроса передаются сводка и
последняя пара реплик. На первом проходе старого чата обрабатывается ограниченный
по размеру хвост из последних 20 завершённых ответов; более ранние сообщения без
готовой сводки автоматически не перечитываются. Полная история остаётся в БД.
Сводка нужна для темы, имён и ограничений; подтверждения фактов по-прежнему берутся
из текущего RAG-поиска. При ошибке суммаризатора используется ограниченная история
и доступная прежняя сводка. Ошибка отображается в диагностике памяти.

В интерфейсе раздел «Сводка истории» показывает текст, фактически переданный модели.
Сводка хранится в `rag_turns.memory_summary` для конкретного чата. При отзыве доступа
к исходным документам она скрывается и больше не используется. Для новой БД примените
`migrations/005_history_summary.sql`; в текущей БД миграция уже применена.

Настройки `LLM_THINKING` и `DEEP_LLM_THINKING` удалены. Запросы Ollama всегда содержат
`think=false`; LM Studio — `reasoning_effort=none` (проверено на текущей Qwen 9B).
Для Qwen через vLLM передаётся
[`chat_template_kwargs.enable_thinking=false`](https://docs.vllm.ai/en/stable/features/reasoning_outputs/).
«Глубокий анализ» остаётся графом поиска и проверок, его шаги выполняются без reasoning.
Обрыв по лимиту токенов не становится успешным ответом; в глубоком режиме доступен
повтор с просьбой дать короткий ответ в пределах настроенного retry limit.

## Ошибка переполнения контекста LM Studio

Если сервер сообщает `n_keep >= n_ctx`, входной запрос не помещается в контекст
загруженной модели. В текущей LM Studio модель загружена с 4096 токенами; для неё
в локальном `.env` установлены `GENERATION_CONTEXT_CHARS=4000`,
`DEEP_CONTEXT_CHARS=4000`, `MEMORY_MAX_CHARS=2000`, `DEEP_MAX_TOKENS=1000`.
После изменения `.env` перезапустите backend. История в БД остаётся целиком,
в запрос модели попадает только ограниченная часть.

Это бюджеты по символам: они не гарантируют точный размер в токенах для любого
текста. Для длинных вопросов и больших ответов увеличьте Context Length при
загрузке модели в LM Studio и затем настройте бюджеты backend. Переменные
`LLM_CONTEXT_TOKENS` и `DEEP_CONTEXT_TOKENS` задают `num_ctx` для Ollama;
они не меняют размер контекста загруженной модели LM Studio.
Backend распознаёт ошибку переполнения в обычных и потоковых запросах, не выводя
текст документов из ответа сервера.

## Глубокий анализ

Режим допускает частичный ответ по близким материалам и осторожные выводы.
Модель отдельно отмечает, что найдено в документах, чем ситуация отличается от
вопроса и что является предположением. Отсутствие точного ответа на все части
вопроса само по себе не вызывает отказ. Диагностика `analysis.validation=partial`
обозначает принятый частичный ответ или предположение; это успешное завершение.
Выдуманные факты, противоречия и неверные цитаты по-прежнему требуют исправления.
Если полезных материалов совсем нет, остаётся сообщение об отсутствии данных.


В веб-чате включите галочку **«Глубокий анализ»** перед отправкой. Справа появятся
этапы, поисковые запросы и результат проверки. Во время проверки черновик ещё не окончательный.
После обновления backend перезагрузите страницу, чтобы загрузился новый JavaScript.

В обоих chat endpoints (`/messages` и `/messages/stream`) поддерживается:

```json
{
  "user_id": 20,
  "message": "Вопрос по документам",
  "client_id": "python-test",
  "request_id": "85b9def9-e695-4eb5-b85e-1b184a1f8705",
  "deep_thinking": true,
  "limit": 5
}
```

Для новой операции генерируйте новый UUID. Без `deep_thinking` работает обычный режим.
Прежние `/ask` и `/ask/stream` остаются простыми stateless endpoints.
При переносе на другую БД примените миграцию от администратора:

```powershell
Get-Content -Raw -Encoding utf8 migrations/004_deep_analysis.sql | docker exec -i testagenticrag-db-1 psql -v ON_ERROR_STOP=1 -U rag_admin -d rag
```

В текущей базе миграция уже применена. Настройки бюджетов находятся в `.env.example`.
При пустом `DEEP_LLM_MODEL` используется основная модель. Для отдельного vLLM-сервера:

```dotenv
DEEP_LLM_PROVIDER=openai
DEEP_LLM_BASE_URL=http://адрес-vllm:8000/v1
DEEP_LLM_MODEL=имя-модели-из-v1-models
DEEP_LLM_API_KEY=
DEEP_MAX_STEPS=12
DEEP_MAX_SEARCHES=3
DEEP_RETRY_LIMIT=1
DEEP_TIMEOUT_SECONDS=240
```

Если указан отдельный DEEP_LLM_BASE_URL, основной API-ключ на него автоматически
не пересылается: при необходимости задайте DEEP_LLM_API_KEY отдельно.
`DEEP_CONTEXT_CHARS` управляет бюджетом текста; `DEEP_CONTEXT_TOKENS` отправляется
как num_ctx в Ollama. Для LM Studio/vLLM размер контекста нужно также настроить на
сервере модели. Увеличение backend-бюджета не увеличивает физический контекст сервера.

По умолчанию `DEEP_STRUCTURED_OUTPUT=true`: используются JSON-схемы
[LM Studio](https://lmstudio.ai/docs/developer/openai-compat/structured-output),
[vLLM](https://docs.vllm.ai/en/latest/features/structured_outputs/) или
[Ollama](https://docs.ollama.com/capabilities/structured-outputs).
Если конкретный совместимый сервер не поддерживает схемы, можно установить false;
проверка формата на backend при этом остаётся. Это не отключает валидацию ответа.

Параллельная проверка глубокого режима:

```powershell
.\.venv\Scripts\python.exe scripts/load_test.py --users 20,21,22 --requests 3 --parallel 3 --deep
```

В SSE `stage` показывает этап, `search` — очередной запрос, `reset` сбрасывает
неподтверждённый черновик, `done` содержит финальный ответ и `analysis`. При исчерпании
бюджета или неудачной проверке итог может быть отказом без источников. Ошибки и отмена
обрабатываются прежними событиями. Подробная архитектура — `ARCHITECTURE.md`.

Импорт файлов из ролевых папок: **INGESTION.md** (команды сверки, загрузки и назначения ролей).

## Веб-чат

Интерфейс доступен на `/`: пользователи, отдельные диалоги, история, SSE-ответы,
источники и отмена. Подробная карта файлов и поток данных: **ARCHITECTURE.md**.

Память диалога включена для chat endpoints. Последние завершённые ответы помогают
превратить «а во втором?» в самостоятельный запрос. Справа отображаются запрос для
поиска и число использованных пар сообщений. Для развёртывания на другой базе
после `001_chats.sql` примените от администратора:

```powershell
Get-Content -Raw -Encoding utf8 migrations/002_memory.sql | docker exec -i testagenticrag-db-1 psql -v ON_ERROR_STOP=1 -U rag_admin -d rag
```

Настройки (значения по умолчанию):

```dotenv
MEMORY_ENABLED=true
MEMORY_MAX_TURNS=6
MEMORY_MAX_CHARS=6000
MEMORY_REWRITE_MAX_TOKENS=384
```

После изменения кода/настроек перезапустите backend и обновите страницу.
Для проверки: в одном чате спросите о двух объектах, затем «а сколько стоит второй?».
Справа должен появиться запрос с названием второго объекта. Новый чат начинает без памяти.
Запросы без ID чата (`/ask`, `/search`) по-прежнему не используют историю.
Существующая LLM обслуживает и rewrite, и генерацию: отдельная модель не нужна,
но для уточнений появляется один дополнительный вызов и соответствующая задержка.

Перед первым запуском на новой базе примените миграцию от администратора:

```powershell
Get-Content -Raw -Encoding utf8 migrations/001_chats.sql | docker exec -i testagenticrag-db-1 psql -v ON_ERROR_STOP=1 -U rag_admin -d rag
```

В текущей локальной базе эта миграция уже применена. В `.env` добавлен отдельный
случайный `DEV_ADMIN_TOKEN`; его нужно ввести в окно подключения веб-чата.
Обычный `DEV_API_TOKEN` остаётся для прежних `/ask` и `/search`.
Токены не передаются в исходниках страницы и не сохраняются браузером.

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
```

Откройте http://127.0.0.1:8000/ и введите `DEV_ADMIN_TOKEN` из `.env`.
Если запускается копия проекта в другой папке, перенесите новые файлы,
примените миграцию к её БД и добавьте новые настройки в её `.env`.

## API для экспериментов с пользователями

Все `/dev` endpoints требуют `Authorization: Bearer <DEV_ADMIN_TOKEN>`:

| Метод | URL | Назначение |
|---|---|---|
| GET | `/dev/users` | Активные пользователи и роли |
| GET | `/dev/config` | Имя модели, провайдер, лимит параллельности (без секретов) |
| GET | `/dev/users/{user_id}/conversations` | Диалоги пользователя |
| POST | `/dev/conversations` | Создать: `{"user_id":1,"title":"Новый диалог"}` |
| GET | `/dev/conversations/{id}/messages?user_id=1` | История с проверкой владельца |
| POST | `/dev/conversations/{id}/messages` | Полный JSON-ответ |
| POST | `/dev/conversations/{id}/messages/stream` | SSE-ответ |
| POST | `/dev/conversations/{id}/cancel` | Отмена: `{"user_id":1,"request_id":"UUID"}` |

Тело отправки сообщения:

```json
{
  "user_id": 1,
  "message": "Вопрос по документу",
  "client_id": "python-test",
  "request_id": "e015e018-940a-4ffa-9655-625ebfbbe54c",
  "limit": 5
}
```

Генерируйте новый `request_id` для каждого нового вопроса. Для повтора одной и той
же операции используйте прежний ID. Один чат не принимает две генерации одновременно
(`409`), разные чаты — принимают. `MAX_CONCURRENT_GENERATIONS=4` задаёт число активных
новых chat-запросов на backend-процесс; заполнение даёт `429`.
Прежние `/ask` не участвуют в лимите экспериментального chat API.

Для vLLM укажите `LLM_PROVIDER=openai`, его `/v1` URL и точное имя модели.
Ограничение последовательной генерацией отсутствует, отдельный batch-endpoint не нужен.

Параллельный тест (создаёт отдельные диалоги и сохраняет результаты в истории):

```powershell
.\.venv\Scripts\python.exe scripts/load_test.py --users 1,2 --requests 8 --parallel 4
```

Подставьте существующие ID пользователей. Токен скрипт читает из `.env`.
Проверки реальной локальной БД с одноразовыми данными:

```powershell
$env:CHAT_DB_TESTS='1'
.\.venv\Scripts\python.exe -m pytest -q tests/test_chat_database.py
Remove-Item Env:CHAT_DB_TESTS
```

Эти тесты используют Docker-контейнер `testagenticrag-db-1`, администратора `rag_admin`
и базу `rag`; создают и удаляют только собственные временные записи.
Название контейнера можно изменить через `CHAT_TEST_CONTAINER`.

FastAPI использует существующую `public.hybrid_search(text, vector, integer)`.
Миграция добавляет таблицы истории; таблицы документов и поисковые индексы не изменяются.
Поиск работает через `rag_reader` с RLS.
Это первый локальный вариант RAG: один поиск и генерация, без автономного агентного цикла.

## Модели

Для генерации по умолчанию используется **Qwen3.5 9B через Ollama**:
эта модель уже обнаружена в локальном `ollama list` (6,6 ГБ).
KV cache и runtime требуют дополнительную память.
Для RTX 4060 Ti 16 ГБ / RAM 32 ГБ это разумная начальная конфигурация;
начните с одного запроса одновременно и контекста 8192 токена.

```powershell
ollama run qwen3.5:9b
```

Выйти из диалога: `/bye`. Ollama должна продолжать обслуживать порт 11434.
Если приложение Ollama не запущено, в отдельном терминале: `ollama serve`.
Альтернатива: `ollama pull qwen3:8b-q4_K_M` и `LLM_MODEL=qwen3:8b-q4_K_M`.

**BGE-M3 не требует vLLM.** Позже можно поднять отдельный сервис на
SentenceTransformers, TEI или vLLM. Backend ожидает совместимый контракт:

```text
POST {EMBEDDING_BASE_URL}/embeddings
{"model":"BAAI/bge-m3","input":["вопрос"]}
→ {"data":[{"embedding":[...1024 чисел...]}]}
```

Адрес embeddings намеренно пустой. `/health` работает без моделей;
`/search` и `/ask` до настройки возвращают 503 с пояснением.
Модель, revision и подготовка текста должны совпадать с загрузчиком документов.
Совпадение размерности само по себе не гарантирует совместимости.

Для языковой модели через **vLLM** переключить `.env`:

```dotenv
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:8001/v1
LLM_MODEL=<имя модели из /v1/models>
LLM_API_KEY=
```

Деплой и подбор квантованной сборки vLLM выполняются отдельно; код поддерживает
`/v1/chat/completions`. Выбор сервера LLM не меняет embeddings и индексы.

Для языковой модели в **LM Studio** используйте `LLM_PROVIDER=lmstudio`,
`LLM_BASE_URL=http://172.19.0.1:5985/v1` и `LLM_MODEL` с точным ID
загруженной языковой модели из `/v1/models`. Этот провайдер также использует
`/chat/completions`. BGE-M3 остаётся только моделью embeddings, её ID задаётся
в `EMBEDDING_MODEL`, а не `LLM_MODEL`. Ollama при такой конфигурации не требуется.

По умолчанию `LLM_THINKING=false`: для LM Studio отправляется
`reasoning_effort=none`, для Ollama — `think=false`. Отключение проверено
на локальной `qwen/qwen3.5-9b`: возвращается финальный ответ без reasoning.
При `LLM_THINKING=true` LM Studio использует свои настройки reasoning;
рассуждения могут израсходовать весь лимит `LLM_MAX_TOKENS` до финального ответа.

## Установка на Windows

Из папки `backend`, нужен работающий Python 3.11+:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
Copy-Item .env.example .env
```

Если `.venv` и `.env` уже созданы при подготовке backend, эти команды повторять
не нужно (особенно копирование поверх `.env`). Проверенные версии пакетов сохранены
в `requirements.lock.txt`; для повторной установки на Windows/Python 3.12:
`python -m pip install -r requirements.lock.txt`, затем `python -m pip install -e . --no-deps`.
Если `py` не видит установленный Python, укажите полный путь к `python.exe`.

В `.env` укажите **пароль rag_reader**, а не административный пароль `postgres`.
База: `localhost:25432/rag`. Если backend запускается в Compose-сети,
используйте имя сервиса `db:5432`. Номер DEV_USER_ID должен существовать и иметь
доступ к документам. Для генерации токена:

```powershell
.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
```

Скопируйте его в `DEV_API_TOKEN`. Запуск:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
```

Откройте http://localhost:8000/docs. В **Authorize** введите DEV_API_TOKEN.
`GET /health` проверяет процесс и наличие настройки embeddings, а не готовность внешних сервисов.

```powershell
Invoke-RestMethod http://localhost:8000/health
$headers = @{ Authorization = 'Bearer ВАШ_DEV_API_TOKEN' }
$body = @{ query = 'Вопрос по документу'; limit = 5 } | ConvertTo-Json
Invoke-RestMethod http://localhost:8000/search -Method Post -Headers $headers -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
$body = @{ question = 'Вопрос по документу'; limit = 5 } | ConvertTo-Json
Invoke-RestMethod http://localhost:8000/ask -Method Post -Headers $headers -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

## Доступ и ограничения

- Только локальная разработка: token сопоставляется с одним DEV_USER_ID на сервере.
  Это не полноценная многопользовательская авторизация. Production-режим отклоняется
  до реализации проверенных токенов и сопоставления auth_subject → app_users.id.
- Клиент и LLM не задают user_id или роли. Все поиски выполняются в транзакции
  с локальным `app.user_id`; соединение освобождается до генерации ответа.
- Backend проверяет, что роль БД не superuser, не BYPASSRLS, не владелец таблиц,
  и RLS включён. Используйте ранее созданные политики на documents и chunks.
- Текст источников передаётся как недоверенные данные. Номера цитат проверяются,
  но истинность утверждений и соответствие цитаты фрагменту автоматически не доказаны.
- Пустая выдача не вызывает LLM. При отсутствии цитат возвращается отказ;
  неизвестный номер цитаты даёт 502. RRF score не является вероятностью ответа.
- Контекст ограничен по символам, не точным токенизатором; при больших таблицах,
  коде или смене модели потребуется настройка бюджета. Индексы и права не изменяются.
- Загрузчик уже существует в другом рабочем каталоге. Здесь он не дублируется.
  Нет кэша и фоновых задач.

## Потоковая генерация

`POST /ask/stream` принимает тот же JSON `{ "question": "...", "limit": 5 }`
и Bearer-токен, что `/ask`. Возвращает `text/event-stream` (SSE):

```text
event: sources
data: {"sources":[{"id":1,"chunk_id":12,"document_id":1,"title":"...","url":"..."}]}

event: delta
data: {"text":"Часть ответа"}

event: done
data: {"answer":"Полный ответ [1]","sources":[...]}
```

`sources` содержит доступные кандидаты, `delta` — предварительный текст,
`done` — окончательный ответ с проверенными номерами цитат и использованными
источниками. Клиент должен заменить черновик значением `done.answer`.
Если цитат нет, финальный ответ может стать отказом, даже если deltas содержали текст.
При ошибке после начала потока приходит `event: error` с `{ "detail": "...", "status": 502 }`;
внешний HTTP-статус уже остаётся 200. Черновик тогда нельзя считать успешным ответом.
Поток без `done` также считается незавершённым. Рассуждения не передаются.
Получение embeddings и поиск выполняются до начала потока.

Для проверки в PowerShell (Swagger может буферизовать вывод):

```powershell
$request = '{"question":"Вопрос по документу","limit":5}'
[IO.File]::WriteAllText((Join-Path $PWD 'request.json'), $request, [Text.UTF8Encoding]::new($false))
curl.exe -N http://127.0.0.1:8000/ask/stream -H "Authorization: Bearer ВАШ_DEV_API_TOKEN" -H "Content-Type: application/json" --data-binary "@request.json"
```

Будущий браузерный интерфейс должен использовать `fetch` с POST и читать
`response.body` через `ReadableStream`/`TextDecoder`, сохраняя незавершённые
SSE-события между порциями. Обычный `EventSource` не поддерживает этот POST и
Bearer-заголовок. За reverse proxy нужно отключить буферизацию SSE.

## Проверки

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Тесты не обращаются к реальным PostgreSQL/моделям. Проверяют API-авторизацию,
невозможность подменить ID через тело запроса, обработку embeddings, оба LLM API,
цитаты, пустой контекст и транзакционную установку пользователя.
Отдельно на реальной БД проверьте запросы двух пользователей с разными ролями,
отзыв доступа и отсутствие переноса контекста между соединениями пула.

Источники: [Ollama Qwen3 8B Q4_K_M](https://ollama.com/library/qwen3:8b-q4_K_M),
[Ollama Chat API](https://docs.ollama.com/api/chat),
[BGE-M3](https://huggingface.co/BAAI/bge-m3),
[vLLM API](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/).
