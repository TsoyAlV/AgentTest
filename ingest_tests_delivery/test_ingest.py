"""Read-only ingestion tests. Run with unittest; PostgreSQL is never used."""
import importlib.util
import os
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import httpx
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get("INGEST_PROJECT_DIR", str(HERE.parent)))
spec = importlib.util.spec_from_file_location("ingest_under_test", PROJECT / "ingest.py")
ingest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest)
SITE = "http://151.243.177.216/"
REAL_CLIENT = httpx.Client


class WordTokenizer:
    """Deterministic offsets for unit tests; not a replacement for BGE tokenizer."""
    def __call__(self, text, add_special_tokens=False, **kwargs):
        offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        ids = list(range(len(offsets)))
        if add_special_tokens:
            ids = [-1] + ids + [-2]
        return {"input_ids": ids, "offset_mapping": offsets}


class NoDatabaseTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(
            ingest.psycopg, "connect",
            side_effect=AssertionError("Tests must never connect to PostgreSQL"),
        ))
        self.enterContext(patch.object(
            ingest, "save_document",
            side_effect=AssertionError("Tests must never save documents"),
        ))


class UrlTests(NoDatabaseTest):
    def test_removes_fragment_and_spaces_preserves_query(self):
        self.assertEqual(
            ingest.normalize_url(f"  {SITE}?lang=ru#about  "),
            SITE + "?lang=ru",
        )

    def test_accepts_https(self):
        self.assertEqual(ingest.normalize_url("https://example.org/a"), "https://example.org/a")

    def test_rejects_invalid_urls(self):
        for value in ("", "example.org", "/relative", "ftp://example.org", "https://"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ingest.normalize_url(value)


class FetchTests(NoDatabaseTest):
    def use_transport(self, handler):
        transport = httpx.MockTransport(handler)
        self.enterContext(patch.object(
            ingest.httpx, "Client",
            side_effect=lambda **kwargs: REAL_CLIENT(transport=transport, **kwargs),
        ))

    def test_follows_redirect_and_decodes_russian(self):
        seen = []
        def handler(request):
            seen.append(request)
            if request.url.path == "/":
                return httpx.Response(302, headers={"location": "/page"})
            return httpx.Response(200, text="<h1>Привет, Ансан</h1>", headers={"content-type": "Text/HTML; charset=utf-8"})
        self.use_transport(handler)
        html, url = ingest.fetch_html(SITE)
        self.assertIn("Привет, Ансан", html)
        self.assertEqual(url, SITE + "page")
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0].headers["user-agent"], "ProjectRAGLoader/1.0")

    def test_http_errors_are_not_treated_as_documents(self):
        for status in (404, 500):
            with self.subTest(status=status):
                with patch.object(ingest.httpx, "Client", side_effect=lambda **kw: REAL_CLIENT(
                    transport=httpx.MockTransport(lambda request: httpx.Response(status)), **kw
                )):
                    with self.assertRaises(httpx.HTTPStatusError):
                        ingest.fetch_html(SITE)

    def test_rejects_non_html_and_missing_content_type(self):
        for content_type in ("application/pdf", "application/json", None):
            headers = {"content-type": content_type} if content_type else {}
            with self.subTest(content_type=content_type):
                with patch.object(ingest.httpx, "Client", side_effect=lambda **kw: REAL_CLIENT(
                    transport=httpx.MockTransport(lambda request: httpx.Response(200, headers=headers, content=b"data")), **kw
                )):
                    with self.assertRaises(ValueError):
                        ingest.fetch_html(SITE)

    def test_timeout_propagates(self):
        def handler(request):
            raise httpx.ReadTimeout("timeout", request=request)
        self.use_transport(handler)
        with self.assertRaises(httpx.ReadTimeout):
            ingest.fetch_html(SITE)


class ExtractionTests(NoDatabaseTest):
    def test_rendered_snapshot_retains_tariff(self):
        from lxml import html as lxml_html
        html = (HERE / "fixtures" / "aceplace_rendered.html").read_text(encoding="utf-8")
        nodes = lxml_html.fromstring(html).xpath('//*[@data-money="weekdayPrice"]')
        self.assertTrue(nodes)
        price = " ".join(nodes[0].text_content().split())
        self.assertTrue(price)
        _, text = ingest.extract_page(html, SITE)
        self.assertTrue(
            price in " ".join(text.split()),
            f"Extraction lost the rendered weekday tariff: {price}",
        )

    def test_site_snapshot_retains_business_facts(self):
        html = (HERE / "fixtures" / "aceplace.html").read_text(encoding="utf-8")
        title, text = ingest.extract_page(html, SITE)
        self.assertTrue(title.strip())
        self.assertIn("AcePlace", text)
        for fact in ("50", "PS5", "Газовые баллоны", "Парковка", "010-7735-1994"):
            with self.subTest(fact=fact):
                self.assertIn(fact, text)
        self.assertNotIn("<script", text)
        self.assertNotIn("<svg", text)
        self.assertNotIn("data-i18n=", text)

    def test_missing_title_falls_back_to_url(self):
        with patch.object(ingest.trafilatura, "extract_metadata", return_value=None), patch.object(
            ingest.trafilatura, "extract", return_value="  Содержимое  "
        ):
            self.assertEqual(ingest.extract_page("html", SITE), (SITE, "Содержимое"))

    def test_empty_extraction_raises(self):
        for value in (None, "", "  \n "):
            with self.subTest(value=value), patch.object(ingest.trafilatura, "extract", return_value=value):
                with self.assertRaises(ValueError):
                    ingest.extract_page("<html></html>", SITE)


class ChunkTests(NoDatabaseTest):
    def setUp(self):
        super().setUp()
        self.tokenizer = WordTokenizer()

    def test_sections_keep_intro_and_replace_sibling(self):
        text = "Введение\n# Главная\nПервый\n## Детали\nВторой\n## Правила\nТретий"
        self.assertEqual(list(ingest.iter_sections(text)), [
            ("", "Введение"), ("Главная", "Первый"),
            ("Главная / Детали", "Второй"), ("Главная / Правила", "Третий"),
        ])

    def test_sibling_h2_without_h1_has_no_false_parent(self):
        self.assertEqual(list(ingest.iter_sections("## Зал\nОписание зала\n## Кухня\nОписание кухни")), [
            ("Зал", "Описание зала"), ("Кухня", "Описание кухни"),
        ])

    def test_empty_sections_not_returned(self):
        self.assertEqual(list(ingest.iter_sections("# Пусто\n## Тоже пусто")), [])

    def test_token_windows_overlap_and_cover_every_word(self):
        words = [f"слово{i}" for i in range(23)]
        with patch.object(ingest, "CHUNK_SIZE", 10), patch.object(ingest, "CHUNK_OVERLAP", 3):
            chunks = list(ingest.split_text(" ".join(words), self.tokenizer))
        self.assertEqual([c.split() for c in chunks], [words[:10], words[7:17], words[14:]])

    def test_prefers_paragraph_boundary(self):
        words = [f"w{i}" for i in range(20)]
        text = " ".join(words[:8]) + "\n\n" + " ".join(words[8:])
        with patch.object(ingest, "CHUNK_SIZE", 10), patch.object(ingest, "CHUNK_OVERLAP", 2):
            chunks = list(ingest.split_text(text, self.tokenizer))
        self.assertEqual(chunks[0].split(), words[:8])
        self.assertEqual(chunks[1].split()[:2], words[6:8])

    def test_empty_text_yields_no_windows(self):
        self.assertEqual(list(ingest.split_text(" \n ", self.tokenizer)), [])

    def test_make_chunks_preserves_section_metadata(self):
        self.assertEqual(ingest.make_chunks("Вступление\n# Зал\nДо 50 гостей", self.tokenizer), [
            {"section_title": None, "content": "Вступление"},
            {"section_title": "Зал", "content": "До 50 гостей"},
        ])

    def test_empty_document_raises(self):
        with self.assertRaises(ValueError):
            ingest.make_chunks("# Только заголовок", self.tokenizer)


class EmbeddingTests(NoDatabaseTest):
    def setUp(self):
        super().setUp()
        self.chunks = [{"section_title": "Кухня", "content": "Можно свою еду"}]
        vector = np.zeros((1, 1024), dtype=np.float32)
        vector[0, 0] = 1
        self.model = SimpleNamespace(
            tokenizer=WordTokenizer(), max_seq_length=8192,
            encode=MagicMock(return_value=vector),
        )

    def test_embeds_title_section_and_body_as_batch(self):
        result = ingest.embed_chunks("AcePlace", self.chunks, self.model)
        self.assertEqual(result.shape, (1, 1024))
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(self.model.encode.call_args.args[0], ["AcePlace\n\nКухня\n\nМожно свою еду"])
        self.assertTrue(self.model.encode.call_args.kwargs["normalize_embeddings"])

    def test_missing_section_does_not_add_none(self):
        ingest.embed_chunks("AcePlace", [{"section_title": None, "content": "Текст"}], self.model)
        self.assertEqual(self.model.encode.call_args.args[0], ["AcePlace\n\nТекст"])

    def test_multiple_chunks_keep_order(self):
        self.model.encode.return_value = np.ones((2, 1024))
        chunks = self.chunks + [{"section_title": "Зал", "content": "До 50 гостей"}]
        result = ingest.embed_chunks("AcePlace", chunks, self.model)
        self.assertEqual(result.shape, (2, 1024))
        self.assertIn("До 50 гостей", self.model.encode.call_args.args[0][1])

    def test_context_limit_checked_before_encoding(self):
        self.model.max_seq_length = 3
        with self.assertRaises(ValueError):
            ingest.embed_chunks("AcePlace", self.chunks, self.model)
        self.model.encode.assert_not_called()

    def test_rejects_invalid_vectors(self):
        bad = {
            "wrong_dimensions": np.ones((1, 768)),
            "wrong_row_count": np.ones((2, 1024)),
            "nan": np.full((1, 1024), np.nan),
            "infinity": np.full((1, 1024), np.inf),
            "zero": np.zeros((1, 1024)),
        }
        for name, vector in bad.items():
            with self.subTest(name=name):
                self.model.encode.return_value = vector
                with self.assertRaises(ValueError):
                    ingest.embed_chunks("AcePlace", self.chunks, self.model)

    def test_get_model_cached_without_downloading(self):
        ingest.get_model.cache_clear()
        try:
            with patch.object(ingest, "SentenceTransformer", return_value=self.model) as factory:
                self.assertIs(ingest.get_model(), ingest.get_model())
                factory.assert_called_once_with(ingest.MODEL_NAME)
        finally:
            ingest.get_model.cache_clear()


class BrowserTests(NoDatabaseTest):
    def test_browser_closed_on_success_http_error_and_timeout(self):
        for outcome in ("success", "404", "timeout"):
            with self.subTest(outcome=outcome):
                manager = MagicMock()
                browser = manager.return_value.__enter__.return_value.chromium.launch.return_value
                page = browser.new_page.return_value
                page.goto.return_value = SimpleNamespace(status=404 if outcome == "404" else 200)
                page.content.return_value = "<main>Готово</main>"
                page.url = SITE
                if outcome == "timeout":
                    page.locator.return_value.first.wait_for.side_effect = TimeoutError("not ready")
                with patch("playwright.sync_api.sync_playwright", manager):
                    if outcome == "success":
                        self.assertEqual(ingest.fetch_browser_html(SITE, "main"), ("<main>Готово</main>", SITE))
                        page.locator.assert_called_once_with("main")
                    else:
                        with self.assertRaises(ValueError if outcome == "404" else TimeoutError):
                            ingest.fetch_browser_html(SITE, "main")
                browser.close.assert_called_once()


@unittest.skipUnless(os.environ.get("RUN_LIVE_TESTS") == "1", "Set RUN_LIVE_TESTS=1 for network tests")
class LiveSiteTests(NoDatabaseTest):
    def test_http_extraction_and_chunking(self):
        html, final_url = ingest.fetch_html(SITE)
        title, text = ingest.extract_page(html, final_url)
        self.assertTrue(title.strip())
        self.assertIn("AcePlace", text)
        self.assertIn("PS5", text)
        self.assertIn("010-7735-1994", text)
        chunks = ingest.make_chunks(text, WordTokenizer())
        self.assertTrue(chunks)
        self.assertTrue(all(c["content"].strip() for c in chunks))


@unittest.skipUnless(os.environ.get("RUN_BROWSER_TESTS") == "1", "Set RUN_BROWSER_TESTS=1 for real Chromium")
class LiveBrowserTests(NoDatabaseTest):
    def test_javascript_populates_tariff(self):
        selector = '[data-money="weekdayPrice"]:not(:empty)'
        html, final_url = ingest.fetch_browser_html(SITE, selector)
        from lxml import html as lxml_html
        nodes = lxml_html.fromstring(html).xpath('//*[@data-money="weekdayPrice"]')
        self.assertTrue(nodes)
        price = nodes[0].text_content().strip()
        self.assertTrue(price)
        _, text = ingest.extract_page(html, final_url)
        self.assertTrue(
            " ".join(price.split()) in " ".join(text.split()),
            f"Extraction lost the rendered weekday tariff: {price}",
        )


@unittest.skipUnless(os.environ.get("RUN_MODEL_TESTS") == "1", "Set RUN_MODEL_TESTS=1 to load real BGE-M3")
class RealModelTests(NoDatabaseTest):
    def test_real_tokenizer_and_1024_dimensional_embeddings(self):
        html = (HERE / "fixtures" / "aceplace.html").read_text(encoding="utf-8")
        title, text = ingest.extract_page(html, SITE)
        model = ingest.get_model()
        chunks = ingest.make_chunks(text, model.tokenizer)
        vectors = ingest.embed_chunks(title, chunks, model)
        self.assertEqual(vectors.shape, (len(chunks), 1024))
        self.assertTrue(np.isfinite(vectors).all())
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
