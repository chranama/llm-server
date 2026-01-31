# integrations/test_full/test_extract_smoke.py
from __future__ import annotations

import pytest

from integrations.lib.assertions import assert_extract_matches_golden
from integrations.lib.fixtures import load_golden_fixture

pytestmark = [pytest.mark.asyncio, pytest.mark.full, pytest.mark.requires_api_key]


async def test_extract_works_for_sroie_receipt_fixture(client, assert_mode):
    assert assert_mode is True

    fx = load_golden_fixture(kind="sroie", name="receipt_001")
    await assert_extract_matches_golden(client=client, fixture=fx)