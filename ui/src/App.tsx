// ui/src/App.tsx
import React, { useMemo, useState } from "react";
import { Playground } from "./components/playground/Playground";
import { AdminPage } from "./components/admin/AdminPage";
import { UiRuntimeConfig } from "./lib/runtime_config";

interface AppProps {
  runtimeConfig: UiRuntimeConfig;
}

type TopTab = "playground" | "admin";

const API_KEY = import.meta.env.VITE_API_KEY as string | undefined;

export default function App({ runtimeConfig }: AppProps) {
  const hasKey = Boolean(API_KEY && API_KEY.trim().length > 0);

  const apiBase = runtimeConfig.api?.base_url?.trim() || "/api";

  const apiBaseNormalized = useMemo(() => {
    return apiBase.endsWith("/") ? apiBase.slice(0, -1) : apiBase;
  }, [apiBase]);

  const [tab, setTab] = useState<TopTab>("playground");

  const TabButton = ({
    label,
    active,
    onClick,
  }: {
    label: string;
    active: boolean;
    onClick: () => void;
  }) => (
    <button
      onClick={onClick}
      style={{
        padding: "0.45rem 0.85rem",
        borderRadius: 999,
        border: active ? "1px solid #2563eb" : "1px solid #cbd5f5",
        background: active ? "#2563eb" : "white",
        color: active ? "white" : "#0f172a",
        fontWeight: 700,
        cursor: "pointer",
      }}
    >
      {label}
    </button>
  );

  return (
    <div style={{ minHeight: "100vh", padding: "1.5rem", fontFamily: "system-ui, sans-serif" }}>
      <header style={{ marginBottom: "1.25rem" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
          <h1 style={{ fontSize: "1.75rem", fontWeight: 800, margin: 0 }}>
            LLM Server
          </h1>

          <span
            title={
              hasKey
                ? "VITE_API_KEY is set in the UI environment."
                : "VITE_API_KEY is missing. Requests to protected endpoints will fail with 401."
            }
            style={{
              fontSize: 12,
              padding: "4px 10px",
              borderRadius: 999,
              border: "1px solid #cbd5f5",
              background: hasKey ? "#ecfdf5" : "#fff7ed",
              color: hasKey ? "#065f46" : "#9a3412",
              fontWeight: 700,
            }}
          >
            {hasKey ? "API key: configured" : "API key: missing"}
          </span>

          <span
            title="Runtime API base loaded from config.json"
            style={{
              fontSize: 12,
              padding: "4px 10px",
              borderRadius: 999,
              border: "1px solid #e2e8f0",
              background: "#f8fafc",
              color: "#334155",
              fontWeight: 700,
            }}
          >
            API base: {apiBaseNormalized}
          </span>
        </div>

        <p style={{ color: "#64748b", marginTop: 10, marginBottom: 10 }}>
          Extraction-first UI for <code>/v1/extract</code> + operational visibility via admin endpoints.
        </p>

        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center", marginBottom: 10 }}>
          <TabButton label="Playground" active={tab === "playground"} onClick={() => setTab("playground")} />
          <TabButton label="Admin" active={tab === "admin"} onClick={() => setTab("admin")} />

          <div style={{ flex: 1 }} />

          <a
            href={`${apiBaseNormalized}/v1/schemas`}
            target="_blank"
            rel="noreferrer"
            style={{ color: "#2563eb", textDecoration: "none", fontWeight: 700 }}
          >
            Schemas
          </a>
          <a
            href={`${apiBaseNormalized}/docs`}
            target="_blank"
            rel="noreferrer"
            style={{ color: "#2563eb", textDecoration: "none", fontWeight: 700 }}
          >
            API docs
          </a>
          <a
            href={`${apiBaseNormalized}/metrics`}
            target="_blank"
            rel="noreferrer"
            style={{ color: "#2563eb", textDecoration: "none", fontWeight: 700 }}
          >
            Metrics
          </a>
        </div>

        {!hasKey && (
          <div
            style={{
              marginTop: 12,
              padding: 12,
              borderRadius: 12,
              border: "1px solid #fed7aa",
              background: "#fff7ed",
              color: "#9a3412",
              whiteSpace: "pre-wrap",
            }}
          >
            <strong>Heads up:</strong> <code>VITE_API_KEY</code> is not set for the UI. Protected endpoints (like{" "}
            <code>/v1/extract</code>, <code>/v1/generate</code>, and admin endpoints) will return 401.
            {"\n"}
            Set it in your UI env (e.g. <code>.env</code>) and restart the UI dev server.
          </div>
        )}
      </header>

      {tab === "playground" ? <Playground /> : <AdminPage />}
    </div>
  );
}