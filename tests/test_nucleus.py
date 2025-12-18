"""Tests for the Nucleus organelle."""

import pytest
import os
from unittest.mock import patch

from operon_ai.providers import MockProvider, LLMResponse


class TestNucleus:
    """Test the Nucleus organelle."""

    def test_nucleus_with_explicit_provider(self):
        """Nucleus should accept an explicit provider."""
        from operon_ai.organelles.nucleus import Nucleus

        mock = MockProvider(responses={"test": "response"})
        nucleus = Nucleus(provider=mock)

        assert nucleus.provider.name == "mock"

    def test_nucleus_auto_detects_provider(self):
        """Nucleus should auto-detect available providers."""
        from operon_ai.organelles.nucleus import Nucleus

        # With no keys, should fall back to mock
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            nucleus = Nucleus()
            assert nucleus.provider.name == "mock"

    def test_nucleus_transcribe(self):
        """Nucleus.transcribe should call provider and return response."""
        from operon_ai.organelles.nucleus import Nucleus, Transcription

        mock = MockProvider(responses={"hello": "world"})
        nucleus = Nucleus(provider=mock)

        result = nucleus.transcribe("hello")

        assert result.content == "world"
        assert len(nucleus.transcription_log) == 1
        assert isinstance(nucleus.transcription_log[0], Transcription)

    def test_nucleus_tracks_energy_cost(self):
        """Nucleus should track ATP cost of transcriptions."""
        from operon_ai.organelles.nucleus import Nucleus

        mock = MockProvider()
        nucleus = Nucleus(provider=mock, base_energy_cost=15)

        result = nucleus.transcribe("test prompt")

        assert nucleus.transcription_log[0].energy_cost == 15

    def test_nucleus_transcription_log_audit_trail(self):
        """Nucleus should maintain complete audit trail."""
        from operon_ai.organelles.nucleus import Nucleus

        mock = MockProvider()
        nucleus = Nucleus(provider=mock)

        nucleus.transcribe("first")
        nucleus.transcribe("second")
        nucleus.transcribe("third")

        assert len(nucleus.transcription_log) == 3
        prompts = [t.prompt for t in nucleus.transcription_log]
        assert prompts == ["first", "second", "third"]
