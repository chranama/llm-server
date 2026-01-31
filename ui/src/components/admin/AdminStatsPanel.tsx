// ui/src/components/admin/AdminStatsPanel.tsx
import React, { useEffect, useState } from "react";
import { ApiError, AdminStatsResponse, adminGetStats } from "../../lib/api";

export function AdminStatsPanel() {
  const [windowDays, setWindowDays] = useState<number>(30);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<AdminStatsResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setErr(null);
    try {
      const r = await adminGetStats(windowDays);
      setData(r);
    } catch (e: any) {
      if (e instanceof ApiError && e.bodyJson) setErr(JSON.stringify(e.bodyJson, null, 2));
      else setErr(e?.message ?? "Failed to load admin stats");
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [windowDays]);

  return (
    <div style={{ border: "1px solid #e2e8f0", borderRadius: 14, padding: 14, background: "white" }}>
      <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 10, flexWrap: "wrap" }}>
        <div style={{ fontWeight: 900 }}>Usage stats</div>

        <label style={{ display: "flex", alignItems: "center", gap: 8, color: "#334155", fontWeight: 700 }}>
          Window (days)
          <input
            type="number"
            value={windowDays}
            min={1}
            max={365}
            onChange={(e) => setWindowDays(Number(e.target.value))}
            style={{
              width: 90,
              padding: "6px 8px",
              borderRadius: 10,
              border: "1px solid #cbd5f5",
            }}
          />
        </label>

        <button
          onClick={load}
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
          {loading ? "Loading…" : "Refresh"}
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
          {loading ? "Loading…" : "No data yet."}
        </div>
      )}

      {data && (
        <>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 12 }}>
            <div><b>Total requests:</b> {data.total_requests}</div>
            <div><b>Prompt tokens:</b> {data.total_prompt_tokens}</div>
            <div><b>Completion tokens:</b> {data.total_completion_tokens}</div>
            <div><b>Avg latency:</b> {data.avg_latency_ms ?? "—"} ms</div>
            <div style={{ color: "#64748b" }}><b>Since:</b> {data.since}</div>
          </div>

          <div style={{ fontWeight: 900, marginBottom: 6 }}>Per-model</div>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr style={{ textAlign: "left" }}>
                  <th style={{ padding: "6px 8px", borderBottom: "1px solid #e2e8f0" }}>Model</th>
                  <th style={{ padding: "6px 8px", borderBottom: "1px solid #e2e8f0" }}>Requests</th>
                  <th style={{ padding: "6px 8px", borderBottom: "1px solid #e2e8f0" }}>Prompt tokens</th>
                  <th style={{ padding: "6px 8px", borderBottom: "1px solid #e2e8f0" }}>Completion tokens</th>
                  <th style={{ padding: "6px 8px", borderBottom: "1px solid #e2e8f0" }}>Avg latency (ms)</th>
                </tr>
              </thead>
              <tbody>
                {data.per_model.map((r) => (
                  <tr key={r.model_id}>
                    <td style={{ padding: "6px 8px", borderBottom: "1px solid #f1f5f9", fontFamily: "monospace" }}>
                      {r.model_id}
                    </td>
                    <td style={{ padding: "6px 8px", borderBottom: "1px solid #f1f5f9" }}>{r.total_requests}</td>
                    <td style={{ padding: "6px 8px", borderBottom: "1px solid #f1f5f9" }}>{r.total_prompt_tokens}</td>
                    <td style={{ padding: "6px 8px", borderBottom: "1px solid #f1f5f9" }}>{r.total_completion_tokens}</td>
                    <td style={{ padding: "6px 8px", borderBottom: "1px solid #f1f5f9" }}>{r.avg_latency_ms ?? "—"}</td>
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