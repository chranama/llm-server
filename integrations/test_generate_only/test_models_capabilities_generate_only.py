# integrations/test_generate_only/test_models_capabilities_generate_only.py
from __future__ import annotations

import pytest

from integrations.lib.assertions import assert_models_generate_only

pytestmark = [pytest.mark.asyncio, pytest.mark.generate_only, pytest.mark.requires_api_key]


async def test_models_endpoint_contract_generate_only(models_snapshot, assert_mode):
    """
    Primary contract: /v1/models says extract is disabled everywhere.
    """
    # assert_mode enforces --mode intent if set; harmless in auto
    assert assert_mode is True
    assert_models_generate_only(models_snapshot)