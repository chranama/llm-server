# integrations/test_full/test_models_capabilities_full.py
from __future__ import annotations

import pytest

from integrations.lib.assertions import assert_models_full

pytestmark = [pytest.mark.asyncio, pytest.mark.full, pytest.mark.requires_api_key]


async def test_models_endpoint_contract_full(models_snapshot, assert_mode):
    assert assert_mode is True
    assert_models_full(models_snapshot)