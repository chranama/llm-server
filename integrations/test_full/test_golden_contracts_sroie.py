# integrations/test_full/test_golden_contracts_sroie.py
from __future__ import annotations

import pytest

from integrations.lib.assertions import assert_extract_matches_golden
from integrations.lib.fixtures import iter_golden_fixtures

pytestmark = [pytest.mark.asyncio, pytest.mark.full, pytest.mark.requires_api_key]


@pytest.mark.parametrize("fixture", list(iter_golden_fixtures(kind="sroie")))
async def test_all_sroie_goldens_pass_contract(client, assert_mode, fixture):
    assert assert_mode is True
    await assert_extract_matches_golden(client=client, fixture=fixture)