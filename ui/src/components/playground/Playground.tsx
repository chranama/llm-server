// ui/src/components/playground/Playground.tsx
import React, { useEffect, useMemo, useState } from "react";
import {
  callExtract,
  callGenerate,
  listSchemas,
  getSchema,
  getCapabilities,
  ApiError,
  SchemaIndexItem,
  JsonSchema,
  CapabilitiesResponseBody,
} from "../../lib/api";

import type { Mode } from "./types";
import { prettyJson, toNumberOr, copyToClipboard } from "./utils";
import { summarizeSchema } from "./schema";
import { buildPerFieldDiff } from "./diff";
import { buildCurlExtract, buildCurlGenerate } from "./curl";

import { ExtractPanel } from "./ExtractPanel";
import { GeneratePanel } from "./GeneratePanel";
import { SchemaInspector } from "./SchemaInspector";

export function Playground() {
  // Default to Generate so we don't flash an Extract UI that may be disabled.
  const [mode, setMode] = useState<Mode>("generate");

  const [loading, setLoading] = useState(false);

  const [extractError, setExtractError] = useState<string | null>(null);
  const [generateError, setGenerateError] = useState<string | null>(null);

  const activeError = mode === "extract" ? extractError : generateError;
  const setActiveError = (msg: string | null) => {
    if (mode === "extract") setExtractError(msg);
    else setGenerateError(msg);
  };

  // -----------------------------
  // Capabilities gating
  // -----------------------------
  const [caps, setCaps] = useState<CapabilitiesResponseBody | null>(null);
  const extractEnabled = caps?.extract === true;

  useEffect(() => {
    let canceled = false;

    async function loadCaps() {
      try {
        const c = await getCapabilities();
        if (canceled) return;

        setCaps({
          generate: Boolean((c as any)?.generate),
          extract: Boolean((c as any)?.extract),
        });
      } catch {
        if (canceled) return;
        // Safe fallback: generate-only UI
        setCaps({ generate: true, extract: false });
      }
    }

    loadCaps();

    return () => {
      canceled = true;
    };
  }, []);

  // If Extract is disabled, never allow the Extract tab to remain selected.
  useEffect(() => {
    if (caps && !extractEnabled && mode === "extract") {
      setMode("generate");
      setExtractError(null);
    }
  }, [caps, extractEnabled, mode]);

  // -----------------------------
  // Schemas (Extract-only)
  // -----------------------------
  const [schemas, setSchemas] = useState<SchemaIndexItem[]>([]);
  const [schemasLoading, setSchemasLoading] = useState(false);

  const schemaOptions = useMemo(() => {
    return [...schemas].sort((a, b) => (a.schema_id || "").localeCompare(b.schema_id || ""));
  }, [schemas]);

  const [schemaJson, setSchemaJson] = useState<JsonSchema | null>(null);
  const [schemaJsonLoading, setSchemaJsonLoading] = useState(false);
  const [schemaJsonError, setSchemaJsonError] = useState<string | null>(null);

  // Extract
  const [schemaId, setSchemaId] = useState<string>("");
  const [extractText, setExtractText] = useState<string>(
    "Company: ACME Corp\nDate: 2024-01-01\nTotal: $12.34\nAddress: 123 Main St, Springfield"
  );
  const [extractOutput, setExtractOutput] = useState<string>("");

  const [extractCache, setExtractCache] = useState<boolean>(true);
  const [extractRepair, setExtractRepair] = useState<boolean>(true);
  const [extractMaxNewTokens, setExtractMaxNewTokens] = useState<number>(512);
  const [extractTemperature, setExtractTemperature] = useState<number>(0.0);

  const [extractDataLatest, setExtractDataLatest] = useState<Record<string, any> | null>(null);
  const [extractDataBaseline, setExtractDataBaseline] = useState<Record<string, any> | null>(null);
  const [autoBaseline, setAutoBaseline] = useState<boolean>(true);
  const [diffShowUnchanged, setDiffShowUnchanged] = useState<boolean>(false);

  // Generate
  const [prompt, setPrompt] = useState("Write a haiku about autumn leaves.");
  const [genOutput, setGenOutput] = useState("");
  const [modelOverride, setModelOverride] = useState<string>("");
  const [genMaxNewTokens, setGenMaxNewTokens] = useState<number>(128);
  const [genTemperature, setGenTemperature] = useState<number>(0.7);

  // Copy feedback
  const [copyMsg, setCopyMsg] = useState<string | null>(null);
  useEffect(() => {
    if (!copyMsg) return;
    const t = window.setTimeout(() => setCopyMsg(null), 1500);
    return () => window.clearTimeout(t);
  }, [copyMsg]);

  async function loadSchemasOnce() {
    // Only load schemas if Extract is enabled
    if (!extractEnabled) return;

    setSchemasLoading(true);
    try {
      const items = await listSchemas();
      setSchemas(items);

      setSchemaId((prev) => {
        if (prev) return prev;
        const sorted = [...items].sort((a, b) => (a.schema_id || "").localeCompare(b.schema_id || ""));
        return sorted.length > 0 ? sorted[0].schema_id : "";
      });
    } catch (e: any) {
      setExtractError(e?.message ?? "Failed to load schemas");
    } finally {
      setSchemasLoading(false);
    }
  }

  // Load schemas once *after* we know extract is enabled
  useEffect(() => {
    if (!caps) return; // wait for capabilities
    if (!extractEnabled) return;

    let canceled = false;
    (async () => {
      if (canceled) return;
      await loadSchemasOnce();
    })();

    return () => {
      canceled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [caps, extractEnabled]);

  // Load schema json when schemaId changes (extract-only)
  useEffect(() => {
    if (!extractEnabled) return;

    let canceled = false;

    async function loadSchemaJson(id: string) {
      if (!id) {
        setSchemaJson(null);
        setSchemaJsonError(null);
        return;
      }

      setSchemaJsonLoading(true);
      setSchemaJsonError(null);

      try {
        const js = await getSchema(id);
        if (canceled) return;
        setSchemaJson(js);
      } catch (e: any) {
        if (canceled) return;
        if (e instanceof ApiError && e.bodyJson) setSchemaJsonError(prettyJson(e.bodyJson));
        else setSchemaJsonError(e?.message ?? "Failed to load schema JSON");
        setSchemaJson(null);
      } finally {
        if (!canceled) setSchemaJsonLoading(false);
      }
    }

    loadSchemaJson(schemaId);

    return () => {
      canceled = true;
    };
  }, [schemaId, extractEnabled]);

  useEffect(() => {
    setExtractDataLatest(null);
    setExtractDataBaseline(null);
  }, [schemaId]);

  const handleRunExtract = async () => {
    if (!extractEnabled) {
      setExtractError("Extraction is disabled in this deployment.");
      return;
    }

    setLoading(true);
    setExtractError(null);
    setExtractOutput("");
    setExtractDataLatest(null);

    try {
      if (!schemaId) throw new Error("No schema selected. Is /v1/schemas returning anything?");

      const res = await callExtract({
        schema_id: schemaId,
        text: extractText,
        cache: extractCache,
        repair: extractRepair,
        max_new_tokens: extractMaxNewTokens,
        temperature: extractTemperature,
        ...(modelOverride.trim() ? { model: modelOverride.trim() } : {}),
      });

      const view = {
        schema_id: res.schema_id,
        model: res.model,
        cached: res.cached,
        repair_attempted: res.repair_attempted,
        data: res.data,
      };

      setExtractOutput(prettyJson(view));
      setExtractDataLatest(res.data ?? {});

      setExtractDataBaseline((prev) => {
        if (autoBaseline && !prev) return res.data ?? {};
        return prev;
      });
    } catch (e: any) {
      if (e instanceof ApiError && e.bodyJson) setExtractError(prettyJson(e.bodyJson));
      else setExtractError(e?.message ?? "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const handleRunGenerate = async () => {
    setLoading(true);
    setGenerateError(null);
    setGenOutput("");

    try {
      const res = await callGenerate({
        prompt,
        max_new_tokens: genMaxNewTokens,
        temperature: genTemperature,
        ...(modelOverride.trim() ? { model: modelOverride.trim() } : {}),
      });

      setGenOutput(
        prettyJson({
          model: res.model,
          cached: res.cached,
          output: res.output,
        })
      );
    } catch (e: any) {
      if (e instanceof ApiError && e.bodyJson) setGenerateError(prettyJson(e.bodyJson));
      else setGenerateError(e?.message ?? "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const TabButton = ({
    label,
    active,
    onClick,
    disabled,
    title,
  }: {
    label: string;
    active: boolean;
    onClick: () => void;
    disabled?: boolean;
    title?: string;
  }) => (
    <button
      onClick={() => {
        if (disabled) return;
        onClick();
        setActiveError(null);
      }}
      disabled={Boolean(disabled)}
      title={title}
      style={{
        padding: "0.4rem 0.8rem",
        borderRadius: 999,
        border: active ? "1px solid #2563eb" : "1px solid #cbd5f5",
        background: active ? "#2563eb" : "white",
        color: active ? "white" : "#0f172a",
        fontWeight: 600,
        cursor: disabled ? "not-allowed" : loading ? "wait" : "pointer",
        opacity: disabled ? 0.55 : 1,
      }}
    >
      {label}
    </button>
  );

  const extractDisabled =
    !extractEnabled || loading || schemasLoading || schemaOptions.length === 0 || !schemaId;

  const schemaSummary = useMemo(() => summarizeSchema(schemaJson), [schemaJson]);

  const handleCopyExtractCurl = async () => {
    const curl = buildCurlExtract({
      schema_id: schemaId || "SCHEMA_ID",
      text: extractText,
      cache: extractCache,
      repair: extractRepair,
      max_new_tokens: extractMaxNewTokens,
      temperature: extractTemperature,
      model: modelOverride.trim() ? modelOverride.trim() : undefined,
    });
    const ok = await copyToClipboard(curl);
    setCopyMsg(ok ? "Copied extract curl" : "Copy failed");
  };

  const handleCopyGenerateCurl = async () => {
    const curl = buildCurlGenerate({
      prompt,
      max_new_tokens: genMaxNewTokens,
      temperature: genTemperature,
      model: modelOverride.trim() ? modelOverride.trim() : undefined,
    });
    const ok = await copyToClipboard(curl);
    setCopyMsg(ok ? "Copied generate curl" : "Copy failed");
  };

  const handleCopySchemaJson = async () => {
    const txt = schemaJson ? prettyJson(schemaJson) : "";
    const ok = await copyToClipboard(txt);
    setCopyMsg(ok ? "Copied schema JSON" : "Copy failed");
  };

  const handleReloadSchemaJson = async () => {
    if (!extractEnabled) return;
    if (!schemaId) return;

    setSchemaJsonLoading(true);
    setSchemaJsonError(null);
    try {
      const js = await getSchema(schemaId);
      setSchemaJson(js);
      setCopyMsg("Schema reloaded");
    } catch (e: any) {
      if (e instanceof ApiError && e.bodyJson) setSchemaJsonError(prettyJson(e.bodyJson));
      else setSchemaJsonError(e?.message ?? "Failed to reload schema JSON");
      setSchemaJson(null);
    } finally {
      setSchemaJsonLoading(false);
    }
  };

  const diffRows = useMemo(() => {
    return buildPerFieldDiff({
      schemaJson,
      baseline: extractDataBaseline,
      latest: extractDataLatest,
    });
  }, [schemaJson, extractDataBaseline, extractDataLatest]);

  const filteredDiffRows = useMemo(() => {
    if (diffShowUnchanged) return diffRows;
    return diffRows.filter((r) => r.status !== "same");
  }, [diffRows, diffShowUnchanged]);

  const diffCounts = useMemo(() => {
    const c = { same: 0, changed: 0, added: 0, removed: 0 };
    for (const r of diffRows) c[r.status] += 1;
    return c;
  }, [diffRows]);

  const handleSetBaseline = () => {
    if (!extractDataLatest) return;
    setExtractDataBaseline(extractDataLatest);
    setCopyMsg("Baseline set");
  };

  const extractTabDisabled = caps ? !extractEnabled : true;
  const extractTabTitle = caps
    ? extractEnabled
      ? ""
      : "Extract is disabled in this deployment."
    : "Loading capabilities…";

  return (
    <div style={{ maxWidth: 1180 }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 16 }}>
        <TabButton
          label={extractTabDisabled ? "Extract (disabled)" : "Extract"}
          active={mode === "extract"}
          onClick={() => setMode("extract")}
          disabled={extractTabDisabled || loading}
          title={extractTabTitle}
        />
        <TabButton
          label="Generate"
          active={mode === "generate"}
          onClick={() => setMode("generate")}
          disabled={loading}
        />

        <div style={{ flex: 1 }} />

        {copyMsg && (
          <span
            style={{
              fontSize: 12,
              padding: "4px 10px",
              borderRadius: 999,
              border: "1px solid #cbd5f5",
              background: "#f1f5f9",
              color: "#0f172a",
              fontWeight: 600,
            }}
          >
            {copyMsg}
          </span>
        )}

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <label style={{ fontWeight: 500, color: "#334155" }}>Model</label>
          <input
            value={modelOverride}
            onChange={(e) => setModelOverride(e.target.value)}
            placeholder="(optional override)"
            style={{
              width: 240,
              padding: "8px 10px",
              borderRadius: 10,
              border: "1px solid #cbd5f5",
              fontFamily: "inherit",
            }}
          />
        </div>
      </div>

      {/* Optional hint when extract is disabled */}
      {caps && !extractEnabled && (
        <div
          style={{
            marginBottom: 12,
            padding: "10px 12px",
            borderRadius: 12,
            border: "1px solid #cbd5f5",
            background: "#f8fafc",
            color: "#0f172a",
          }}
        >
          <div style={{ fontWeight: 700, marginBottom: 4 }}>Extraction disabled</div>
          <div style={{ fontSize: 13, color: "#334155" }}>
            This deployment has <span style={{ fontWeight: 700 }}>extract disabled</span>. Generate remains available.
          </div>
        </div>
      )}

      {mode === "extract" && extractEnabled && (
        <div style={{ display: "grid", gridTemplateColumns: "1.25fr 0.85fr", gap: 16, alignItems: "start" }}>
          <ExtractPanel
            loading={loading}
            schemaId={schemaId}
            setSchemaId={setSchemaId}
            schemasLoading={schemasLoading}
            schemaOptions={schemaOptions}
            onReloadSchemas={loadSchemasOnce}
            extractCache={extractCache}
            setExtractCache={setExtractCache}
            extractRepair={extractRepair}
            setExtractRepair={setExtractRepair}
            extractMaxNewTokens={extractMaxNewTokens}
            setExtractMaxNewTokens={setExtractMaxNewTokens}
            extractTemperature={extractTemperature}
            setExtractTemperature={setExtractTemperature}
            extractText={extractText}
            setExtractText={setExtractText}
            extractDisabled={extractDisabled}
            onRunExtract={handleRunExtract}
            onCopyCurl={handleCopyExtractCurl}
            activeError={activeError}
            extractOutput={extractOutput}
            diffRows={diffRows}
            filteredDiffRows={filteredDiffRows}
            diffCounts={diffCounts}
            diffShowUnchanged={diffShowUnchanged}
            setDiffShowUnchanged={setDiffShowUnchanged}
            autoBaseline={autoBaseline}
            setAutoBaseline={setAutoBaseline}
            canSetBaseline={Boolean(extractDataLatest)}
            onSetBaseline={handleSetBaseline}
            canClearBaseline={Boolean(extractDataBaseline)}
            onClearBaseline={() => {
              setExtractDataBaseline(null);
              setCopyMsg("Baseline cleared");
            }}
            schemaJson={schemaJson}
            toNumberOr={toNumberOr}
          />

          <SchemaInspector
            schemaId={schemaId}
            schemaJson={schemaJson}
            schemaJsonLoading={schemaJsonLoading}
            schemaJsonError={schemaJsonError}
            schemaSummary={schemaSummary}
            loading={loading}
            onReload={handleReloadSchemaJson}
            onCopyJson={handleCopySchemaJson}
          />
        </div>
      )}

      {mode === "generate" && (
        <GeneratePanel
          loading={loading}
          prompt={prompt}
          setPrompt={setPrompt}
          genMaxNewTokens={genMaxNewTokens}
          setGenMaxNewTokens={setGenMaxNewTokens}
          genTemperature={genTemperature}
          setGenTemperature={setGenTemperature}
          onCopyCurl={handleCopyGenerateCurl}
          onRun={handleRunGenerate}
          activeError={activeError}
          genOutput={genOutput}
          toNumberOr={toNumberOr}
        />
      )}
    </div>
  );
}