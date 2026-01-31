// ui/src/lib/__tests__/api.admin.test.ts
import { describe, it, expect, vi, beforeEach } from "vitest";
import { adminGetStats, adminListLogs, setApiBaseUrl } from "../api";

describe("api admin endpoints", () => {
  beforeEach(() => {
    setApiBaseUrl("/api");
    vi.restoreAllMocks();
  });

  it("adminGetStats uses window_days query param", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response(JSON.stringify({ ok: true }), { status: 200 }) as any);

    await adminGetStats(7);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];

    expect(String(url)).toContain("/api/v1/admin/stats?");
    expect(String(url)).toContain("window_days=7");
    expect((init as any).method).toBe("GET");
  });

  it("adminListLogs defaults limit=50 offset=0 and omits empty filters", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response(JSON.stringify({ ok: true }), { status: 200 }) as any);

    await adminListLogs({});

    const [url] = fetchMock.mock.calls[0];
    const s = String(url);

    expect(s).toContain("/api/v1/admin/logs?");
    expect(s).toContain("limit=50");
    expect(s).toContain("offset=0");

    expect(s).not.toContain("model_id=");
    expect(s).not.toContain("route=");
    expect(s).not.toContain("api_key=");
    expect(s).not.toContain("from_ts=");
    expect(s).not.toContain("to_ts=");
  });

  it("adminListLogs includes provided filters and custom pagination", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response(JSON.stringify({ ok: true }), { status: 200 }) as any);

    await adminListLogs({
      model_id: "m1",
      route: "/v1/extract",
      api_key: "k",
      from_ts: "2026-01-01T00:00:00Z",
      to_ts: "2026-01-02T00:00:00Z",
      limit: 10,
      offset: 20,
    });

    const [url] = fetchMock.mock.calls[0];
    const s = String(url);

    expect(s).toContain("model_id=m1");
    expect(s).toContain("route=%2Fv1%2Fextract");
    expect(s).toContain("api_key=k");
    expect(s).toContain("from_ts=2026-01-01T00%3A00%3A00Z");
    expect(s).toContain("to_ts=2026-01-02T00%3A00%3A00Z");
    expect(s).toContain("limit=10");
    expect(s).toContain("offset=20");
  });
});