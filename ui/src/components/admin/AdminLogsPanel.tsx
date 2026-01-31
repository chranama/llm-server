// ui/src/components/admin/AdminLogsPanel.tsx
import React, { useEffect, useState } from "react";
import { ApiError, AdminLogsPage, adminListLogs } from "../../lib/api";

export function AdminLogsPanel() {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<AdminLogsPage | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const [modelId, setModelId] = useState("");
  const [route, setRoute] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [limit, setLimit] = useState(50);
  const [offset, setOffset] = useState(0);

  const load = async () => {
    setLoading(true);
    setErr(null);
    try {
      const r = await adminListLogs({
        model_id: modelId.trim() || undefined,
        route: route.trim() || undefined,
        api_key: apiKey.trim() || undefined,
        limit,
        offset,
      });
      setData(r);
    } catch (e: any) {
      if (e instanceof ApiError && e.bodyJson) setErr(JSON.stringify(e.bodyJson, null, 2));
      else setErr(e?.message ?? "Failed to load logs");
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [offset, limit]);

  const canPrev = offset > 0;
  const canNext = data ? offset + limit < data.total : false;

  return (
    <div style={{ border: "1px solid #e2e8f0", borderRadius: 14, padding: 14, background: "white" }}>
      <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 10, flexWrap: "wrap" }}>
        <div style={{ fontWeight: 900 }}>Inference logs</div>

        <input
          value={modelId}
          onChange={(e) => setModelId(e.target.value)}
          placeholder="model_id filter"
          style={{ width: 260, padding: "6px 8px", borderRadius: 10, border: "1px solid #cbd5f5" }}
        />
        <input
          value={route}
          onChange={(e) => setRoute(e.target.value)}
          placeholder="route filter (e.g. /v1/extract)"
          style={{ width: 220, padding: "6px 8px", borderRadius: 10, border: "1px solid #cbd5f5" }}
        />
        <input
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder="api_key filter"
          style={{ width: 220, padding: "6px 8px", borderRadius: 10, border: "1px solid #cbd5f5" }}
        />

        <label style={{ display: "flex", alignItems: "center", gap: 8, color: "#334155", fontWeight: 700 }}>
          Limit
          <input
            type="number"
            value={limit}
            min={1}
            max={200}
            onChange={(e) => setLimit(Number(e.target.value))}
            style={{ width: 80, padding: "6px 8px", borderRadius: 10, border: "1px solid #cbd5f5" }}
          />
        </label>

        <button
          onClick={() => {
            setOffset(0);
            load();
          }}
          disabled={loading}
          style={{
            padding: "0.35rem 0.7rem",
            borderRadius: 10,
            border: "1px solid #cbd5f5",
            background: "white",
            cursor: loading ? "wait" : "pointer",
            fontWeight: 800,
          }}
        >
          {loading ? "Loading…" : "Apply filters"}
        </button>

        <div style={{ flex: 1 }} />

        <button
          onClick={() => setOffset(Math.max(0, offset - limit))}
          disabled={loading || !canPrev}
          style={{
            padding: "0.35rem 0.7rem",
            borderRadius: 10,
            border: "1px solid #cbd5f5",
            background: "white",
            cursor: loading ? "wait" : canPrev ? "pointer" : "not-allowed",
            fontWeight: 800,
            opacity: canPrev ? 1 : 0.55,
          }}
        >
          Prev
        </button>
        <button
          onClick={() => setOffset(offset + limit)}
          disabled={loading || !canNext}
          style={{
            padding: "0.35rem 0.7rem",
            borderRadius: 10,
            border: "1px solid #cbd5f5",
            background: "white",
            cursor: loading ? "wait" : canNext ? "pointer" : "not-allowed",
            fontWeight: 800,
            opacity: canNext ? 1 : 0.55,
          }}
        >
          Next
        </button>
      </div>

      {err && (
        <pre
          style={{
            padding: 12,
            borderRadius: 12,
            border: "1px solid #fecaca",
            background: "#fff1f2",
            color: "#991b1b",
            whiteSpace: "pre-wrap",
          }}
        >
          {err}
        </pre>
      )}

      {!err && !data && (
        <div style={{ color: "#64748b", fontSize: 13 }}>
          {loading ? "Loading…" : "No data."}
        </div>
      )}

      {data && (
        <>
          <div style={{ marginBottom: 8, color: "#64748b", fontSize: 13 }}>
            Total: {data.total} · Showing {data.items.length} · Offset {data.offset}
          </div>

          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr style={{ textAlign: "left" }}>
                  <th style={{ padding: "6px 8px", borderBottom: "1px solid #e2e8f0" }}>Time</th>
                  <th style={{ padding: "6px 8px", borderBottom: "1px solid #e2e8f0" }}>Route</th>
                  <th style={{ padding: "6px 8px", borderBottom: "1px solid #e2e8f0" }}>Model</th>
                  <th style={{ padding: "6px 8px", borderBottom: "1px solid #e2e8f0" }}>Latency</th>
                  <th style={{ padding: "6px 8px", borderBottom: "1px solid #e2e8f0" }}>Tokens</th>
                  <th style={{ padding: "6px 8px", borderBottom: "1px solid #e2e8f0" }}>Key</th>
                  <th style={{ padding: "6px 8px", borderBottom: "1px solid #e2e8f0" }}>Prompt</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((r) => (
                  <tr key={r.id}>
                    <td style={{ padding: "6px 8px", borderBottom: "1px solid #f1f5f9", whiteSpace: "nowrap" }}>
                      {r.created_at}
                    </td>
                    <td style={{ padding: "6px 8px", borderBottom: "1px solid #f1f5f9", fontFamily: "monospace" }}>
                      {r.route}
                    </td>
                    <td style={{ padding: "6px 8px", borderBottom: "1px solid #f1f5f9", fontFamily: "monospace" }}>
                      {r.model_id}
                    </td>
                    <td style={{ padding: "6px 8px", borderBottom: "1px solid #f1f5f9" }}>
                      {r.latency_ms ?? "—"} ms
                    </td>
                    <td style={{ padding: "6px 8px", borderBottom: "1px solid #f1f5f9" }}>
                      p:{r.prompt_tokens ?? "—"} / c:{r.completion_tokens ?? "—"}
                    </td>
                    <td style={{ padding: "6px 8px", borderBottom: "1px solid #f1f5f9", fontFamily: "monospace" }}>
                      {r.api_key ? r.api_key.slice(0, 8) + "…" : "—"}
                    </td>
                    <td style={{ padding: "6px 8px", borderBottom: "1px solid #f1f5f9" }}>
                      <details>
                        <summary style={{ cursor: "pointer", color: "#2563eb", fontWeight: 800 }}>
                          show
                        </summary>
                        <pre style={{ whiteSpace: "pre-wrap", marginTop: 8 }}>
                          {r.prompt}
                        </pre>
                      </details>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}