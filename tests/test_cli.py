"""Tests for src/cli.py (Phase 8 REPL) -- no real Gemini calls made.

All generate_answer invocations are mocked so the test suite can run
entirely offline.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest

from src.cli import _format_citations, _run_repl, main

# ---------------------------------------------------------------------------
# Shared mock payloads
# ---------------------------------------------------------------------------
MOCK_RESULT_EMR = {
    "answer": "Amazon EMR is a cloud big-data platform (Amazon EMR, page 22).",
    "citations": [
        {"id": "service:analytics:amazon-emr", "citation_label": "Amazon EMR, page 22", "chunk_type": "service"},
    ],
    "chunks_used": ["service:analytics:amazon-emr"],
    "retrieved_chunks": [
        {
            "id": "service:analytics:amazon-emr",
            "chunk_type": "service",
            "category": "Analytics",
            "service_name": "Amazon EMR",
            "page_start": 22,
            "page_end": 22,
            "text": "Amazon EMR\n\nAmazon EMR is the industry-leading cloud big data solution...",
        }
    ],
    "timings": {"total_s": 1.25},
}


MOCK_RESULT_MULTI = {
    "answer": "AWS offers Athena, EMR, and Redshift.",
    "citations": [
        {"id": "service:analytics:amazon-athena", "citation_label": "Amazon Athena, page 21", "chunk_type": "service"},
        {"id": "service:analytics:amazon-redshift", "citation_label": "Amazon Redshift, page 25", "chunk_type": "service"},
    ],
    "chunks_used": ["service:analytics:amazon-athena", "service:analytics:amazon-redshift"],
}

MOCK_RESULT_REFUSAL = {
    "answer": "This isn't covered in the document.",
    "citations": [],
    "chunks_used": [],
}


# ---------------------------------------------------------------------------
# Unit tests: _format_citations
# ---------------------------------------------------------------------------
class TestFormatCitations:
    def test_empty_list_returns_empty_string(self):
        assert _format_citations([]) == ""

    def test_single_citation(self):
        assert _format_citations([{"citation_label": "Amazon EMR, page 22"}]) == \
            "Sources: Amazon EMR, page 22"

    def test_multiple_citations_joined_with_semicolon(self):
        cits = [
            {"citation_label": "Amazon Athena, page 21"},
            {"citation_label": "Amazon Redshift, page 25"},
        ]
        assert _format_citations(cits) == \
            "Sources: Amazon Athena, page 21; Amazon Redshift, page 25"

    def test_skips_empty_labels(self):
        cits = [{"citation_label": ""}, {"citation_label": "Amazon EMR, page 22"}, {}]
        assert _format_citations(cits) == "Sources: Amazon EMR, page 22"


# ---------------------------------------------------------------------------
# Integration tests: _run_repl
# ---------------------------------------------------------------------------
class TestRunRepl:
    def _run_with_input(self, input_lines, mock_generate=None):
        side_effects = input_lines + [EOFError()]
        gen_mock = mock_generate or MagicMock(return_value=MOCK_RESULT_EMR)
        captured = io.StringIO()
        with patch("builtins.input", side_effect=side_effects), \
             patch("src.cli.generate_answer", gen_mock), \
             patch("sys.stdout", captured):
            _run_repl()
        return captured.getvalue(), gen_mock

    def test_exit_command_stops_repl(self):
        out, _ = self._run_with_input(["exit"])
        assert "Goodbye!" in out

    def test_quit_command_stops_repl(self):
        out, _ = self._run_with_input(["quit"])
        assert "Goodbye!" in out

    def test_eof_stops_repl_cleanly(self):
        captured = io.StringIO()
        mock_gen = MagicMock()
        with patch("builtins.input", side_effect=EOFError()), \
             patch("src.cli.generate_answer", mock_gen), \
             patch("sys.stdout", captured):
            _run_repl()
        assert "Goodbye!" in captured.getvalue()
        mock_gen.assert_not_called()

    def test_empty_input_does_not_call_generate_answer(self):
        _, mock_gen = self._run_with_input(["", "", "exit"])
        mock_gen.assert_not_called()

    def test_question_calls_generate_answer_once(self):
        _, mock_gen = self._run_with_input(["What is Amazon EMR?", "exit"])
        mock_gen.assert_called_once_with("What is Amazon EMR?")

    def test_answer_and_citations_printed(self):
        out, _ = self._run_with_input(["What is Amazon EMR?", "exit"])
        assert "Amazon EMR is a cloud big-data platform" in out
        assert "Sources: Amazon EMR, page 22" in out

    def test_refusal_answer_no_sources_line(self):
        out, _ = self._run_with_input(
            ["What is Google BigQuery?", "exit"],
            mock_generate=MagicMock(return_value=MOCK_RESULT_REFUSAL),
        )
        assert "This isn't covered in the document." in out
        assert "Sources:" not in out

    def test_query_counter_increments(self):
        out, _ = self._run_with_input(["Question 1", "Question 2", "exit"])
        assert "[Query 1 this session]" in out
        assert "[Query 2 this session]" in out

    def test_rate_limit_error_shows_friendly_message(self):
        mock_gen = MagicMock(side_effect=Exception("RESOURCE_EXHAUSTED: quota exceeded"))
        out, _ = self._run_with_input(["What is Lambda?", "exit"], mock_generate=mock_gen)
        assert "Rate limit hit" in out
        assert "Traceback" not in out
        assert "[Query 1 this session]" not in out   # counter not incremented on error

    def test_unexpected_error_shows_clean_message(self):
        mock_gen = MagicMock(side_effect=RuntimeError("Something broke"))
        out, _ = self._run_with_input(["What is S3?", "exit"], mock_generate=mock_gen)
        assert "An error occurred" in out
        assert "Traceback" not in out

    def test_multiple_citations_formatted_with_semicolons(self):
        out, _ = self._run_with_input(
            ["List some analytics services", "exit"],
            mock_generate=MagicMock(return_value=MOCK_RESULT_MULTI),
        )
        assert "Sources: Amazon Athena, page 21; Amazon Redshift, page 25" in out

    def test_retrieved_chunks_printed_in_repl(self):
        out, _ = self._run_with_input(["What is EMR?", "exit"])
        assert "RETRIEVED CHUNKS SENT TO LLM" in out
        assert "service:analytics:amazon-emr" in out
        assert "Total Duration:" in out



# ---------------------------------------------------------------------------
# main() / --help check
# ---------------------------------------------------------------------------
class TestMain:
    def test_help_flag_exits_zero(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_no_args_starts_repl_then_exits_on_eof(self):
        with patch("builtins.input", side_effect=EOFError()), \
             patch("sys.stdout", io.StringIO()):
            result = main([])
        assert result == 0


# ---------------------------------------------------------------------------
# Unit tests: _clean_llm_response_text
# ---------------------------------------------------------------------------
class TestCleanLLMResponseText:
    def test_plain_string_remains_unchanged(self):
        from src.generation.rag_chain import _clean_llm_response_text
        raw = "  Amazon EC2 is a compute service (Amazon EC2, page 10).  "
        assert _clean_llm_response_text(raw) == "Amazon EC2 is a compute service (Amazon EC2, page 10)."

    def test_list_of_dict_content_blocks_extracted(self):
        from src.generation.rag_chain import _clean_llm_response_text
        raw = [
            {
                "type": "text",
                "text": "Amazon EC2 is a web service that provides secure, resizable compute capacity in the cloud.",
                "extras": {"signature": "Et0PCtoP...blob..."},
            }
        ]
        assert _clean_llm_response_text(raw) == "Amazon EC2 is a web service that provides secure, resizable compute capacity in the cloud."

    def test_multiple_content_parts(self):
        from src.generation.rag_chain import _clean_llm_response_text
        raw = [
            {"type": "text", "text": "Part 1. "},
            {"type": "text", "text": "Part 2."},
        ]
        assert _clean_llm_response_text(raw) == "Part 1. Part 2."

