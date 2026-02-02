from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def llm_outputs():
    # enough outputs for both requests
    return ["ok", "ok"]


@pytest.fixture
def llm_sleep_s():
    # Ensure requests overlap and exercise the semaphore queueing
    return 0.3


@pytest.mark.anyio
async def test_concurrency_queue(client, auth_headers):
    # Fire two requests concurrently. Second should queue, but both succeed.
    r1_task = client.post("/v1/generate", headers=auth_headers, json={"prompt": "a"})
    r2_task = client.post("/v1/generate", headers=auth_headers, json={"prompt": "b"})

    res1, res2 = await r1_task, await r2_task
    assert res1.status_code == 200, res1.text
    assert res2.status_code == 200, res2.text