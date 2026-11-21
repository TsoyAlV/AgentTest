# Тесты ingest.py без PostgreSQL

Запускать из C:\Users\User\Desktop\work_dir\TestAgenticRAG.
Дополнительная установка pytest не нужна: используется unittest.

## Обычный запуск

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s tests -p "test_ingest.py" -v
```

Обычные тесты используют сохранённый HTML сайта http://151.243.177.216/,
httpx.MockTransport, тестовый токенизатор и подставную модель embeddings.
Токенизатор в unit-тестах проверяет алгоритм окон и перекрытия, но не качество
токенизации BGE. Модель не скачивается, запросы к сети не выполняются.
Для браузерных unit-тестов нужен установленный пакет playwright; сам Chromium
для этих unit-тестов не запускается.

В каждом тесте psycopg.connect и save_document заменены функциями,
которые вызывают ошибку при обращении. ingest() и save_document() не тестируются.
Исходный ingest.py не изменяется. Обычный импорт модуля вызывает его load_dotenv().

## Реальный HTTP и Chromium

```powershell
$env:RUN_LIVE_TESTS = "1"
$env:RUN_BROWSER_TESTS = "1"
.\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s tests -p "test_ingest.py" -v
Remove-Item Env:RUN_LIVE_TESTS
Remove-Item Env:RUN_BROWSER_TESTS
```

Для Chromium требуется ранее установленный браузер Playwright:

```powershell
.\.venv\Scripts\python.exe -m playwright install chromium
```

## Реальная модель BGE-M3 (отдельно)

```powershell
$env:RUN_MODEL_TESTS = "1"
.\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s tests -p "test_ingest.py" -v
Remove-Item Env:RUN_MODEL_TESTS
```

Эта проверка использует настоящий токенизатор, вычисляет embeddings из HTML-снимка,
проверяет размерность 1024, конечность значений и норму векторов. При отсутствии
модели в кэше SentenceTransformer скачает её. Проверка сходства по смыслу не входит
в эти тесты. Для воспроизводимости модели зафиксируйте её revision отдельно.

## Обнаруженные проблемы исходного кода

Тесты намеренно проверяют требуемое поведение и не помечают найденные ошибки
как ожидаемые успешные результаты.

1. test_sibling_h2_without_h1_has_no_false_parent: у двух соседних заголовков
   `## Зал` и `## Кухня` второй ошибочно получает путь `Зал / Кухня`.
2. test_rendered_snapshot_retains_tariff: цена буднего дня присутствует в DOM
   после JavaScript, но extract_page() теряет её при извлечении основного текста.
   Это воспроизводится без сети на fixtures/aceplace_rendered.html.
3. При включении RUN_BROWSER_TESTS та же потеря цены проверяется на живом сайте.

fixtures/aceplace.html — исходный HTTP HTML, fixtures/aceplace_rendered.html —
HTML после выполнения JavaScript. Снимки получены 12 ноября 2026 года.
Содержимое живого сайта может измениться, тогда live-тесты требуют пересмотра.
