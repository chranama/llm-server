// ui/src/App.tsx
import React from "react";
import { Playground } from "./components/Playground";
import { UiRuntimeConfig } from "./lib/runtime_config";

interface AppProps {
  runtimeConfig: UiRuntimeConfig;
}

const API_KEY = import.meta.env.VITE_API_KEY as string | undefined;

export default function App({ runtimeConfig }: AppProps) {
  const hasKey = Boolean(API_KEY && API_KEY.trim().length > 0);

  const apiBase =
    runtimeConfig.api?.base_url?.trim() || "/api";

  // Normalize for links
  const apiBaseNormalized =
    apiBase.endsWith("/") ? apiBase.slice(0, -1) : apiBase;

  return (
    <div style={{ minHeight: "100vh", padding: "1.5rem", fontFamily: "system-ui, sans-serif" }}>
      <header style={{ marginBottom: "1.5rem" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
          <h1 style={{ fontSize: "1.75rem", fontWeight: 700, margin: 0 }}>
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
              fontWeight: 600,
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
              fontWeight: 600,
            }}
          >
            API base: {apiBaseNormalized}
          </span>
        </div>

        <p style={{ color: "#64748b", marginTop: 10, marginBottom: 10 }}>
          Extraction-first playground for <code>/v1/extract</code>, with a secondary tab for{" "}
          <code>/v1/generate</code>.
        </p>

        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
          <a
            href={`${apiBaseNormalized}/v1/schemas`}
            target="_blank"
            rel="noreferrer"
            style={{ color: "#2563eb", textDecoration: "none", fontWeight: 600 }}
          >
            View schemas
          </a>
          <a
            href={`${apiBaseNormalized}/docs`}
            target="_blank"
            rel="noreferrer"
            style={{ color: "#2563eb", textDecoration: "none", fontWeight: 600 }}
          >
            API docs
          </a>
          <a
            href={`${apiBaseNormalized}/metrics`}
            target="_blank"
            rel="noreferrer"
            style={{ color: "#2563eb", textDecoration: "none", fontWeight: 600 }}
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
            <code>/v1/extract</code> and <code>/v1/generate</code>) will return 401.
            {"\n"}
            Set it in your UI env (e.g. <code>.env</code>) and restart the UI build/dev server.
          </div>
        )}
      </header>

      <Playground />
    </div>
  );
}