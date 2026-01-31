// ui/src/components/admin/AdminPage.tsx
import React, { useEffect, useState } from "react";
import { ApiError, adminLoadModel } from "../../lib/api";
import { AdminStatsPanel } from "./AdminStatsPanel";
import { AdminLogsPanel } from "./AdminLogsPanel";

export function AdminPage() {
  const [loadMsg, setLoadMsg] = useState<string | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);

  useEffect(() => {
    // clear transient messages
    if (!loadMsg && !loadErr) return;
    const t = window.setTimeout(() => {
      setLoadMsg(null);
      setLoadErr(null);
    }, 2500);
    return () => window.clearTimeout(t);
  }, [loadMsg, loadErr]);

  const handleLoadModel = async () => {
    setLoadErr(null);
    setLoadMsg(null);
    try {
      const r = await adminLoadModel({});
      setLoadMsg(r.already_loaded ? "Model already loaded" : "Model load triggered");
    } catch (e: any) {
      if (e instanceof ApiError && e.bodyJson) setLoadErr(JSON.stringify(e.bodyJson, null, 2));
      else setLoadErr(e?.message ?? "Failed to call /v1/admin/models/load");
    }
  };

  return (
    <div style={{ maxWidth: 1180 }}>
      <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 14, flexWrap: "wrap" }}>
        <div style={{ fontWeight: 900, fontSize: 16 }}>Admin</div>

        <button
          onClick={handleLoadModel}
          style={{
            padding: "0.45rem 0.8rem",
            borderRadius: 10,
            border: "1px solid #cbd5f5",
            background: "white",
            cursor: "pointer",
            fontWeight: 800,
          }}
          title="Dev convenience: ensure weights are loaded"
        >
          Load model
        </button>

        {loadMsg && (
          <span
            style={{
              fontSize: 12,
              padding: "4px 10px",
              borderRadius: 999,
              border: "1px solid #cbd5f5",
              background: "#ecfdf5",
              color: "#065f46",
              fontWeight: 800,
            }}
          >
            {loadMsg}
          </span>
        )}
      </div>

      {loadErr && (
        <pre
          style={{
            padding: 12,
            borderRadius: 12,
            border: "1px solid #fecaca",
            background: "#fff1f2",
            color: "#991b1b",
            whiteSpace: "pre-wrap",
            marginBottom: 14,
          }}
        >
          {loadErr}
        </pre>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 16 }}>
        <AdminStatsPanel />
        <AdminLogsPanel />
      </div>
    </div>
  );
}