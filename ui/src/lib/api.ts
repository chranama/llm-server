// ui/src/lib/api.ts

// -----------------------------
// Types: /v1/generate
// -----------------------------
export interface GenerateRequestBody {
  prompt: string;
  max_new_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  stop?: string[];
  model?: string;
  cache?: boolean; // backend supports cache on /v1/generate
}

export interface GenerateResponseBody {
  model: string;
  output: string;
  cached: boolean;
}

// -----------------------------
// Types: /v1/extract
// -----------------------------
export interface ExtractRequestBody {
  schema_id: string;
  text: string;

  model?: string;

  max_new_tokens?: number;
  temperature?: number;

  cache?: boolean;
  repair?: boolean;
}

export interface ExtractResponseBody {
  schema_id: string;
  model: string;
  data: Record<string, any>;
  cached: boolean;
  repair_attempted: boolean;
}

// -----------------------------
// Types: /v1/schemas
// -----------------------------
export interface SchemaIndexItem {
  schema_id: string;
  title?: string;
  description?: string;
}

// -----------------------------
// Types: /v1/schemas/{schema_id}
// -----------------------------
export type JsonSchema = Record<string, any>;

// -----------------------------
// Types: /v1/models (replaces /v1/capabilities)
// -----------------------------
export interface ModelInfo {
  id: string;
  default: boolean;
  backend?: string | null;
  capabilities?: Record<string, boolean> | null;
  load_mode?: string | null;
  loaded?: boolean | null;
}

export interface ModelsResponseBody {
  default_model: string;
  models: ModelInfo[];
  deployment_capabilities: {
    generate: boolean;
    extract: boolean;
  };
}

// Keep this name so callers don’t have to change imports/usages.
// Now sourced from /v1/models.deployment_capabilities.
export interface CapabilitiesResponseBody {
  generate: boolean;
  extract: boolean;
}

// -----------------------------
// Error type
// -----------------------------
export class ApiError extends Error {
  status: number;
  bodyText: string;
  bodyJson: any | null;

  constructor(status: number, bodyText: string, bodyJson: any | null = null) {
    super(ApiError.makeMessage(status, bodyText, bodyJson));
    this.name = "ApiError";
    this.status = status;
    this.bodyText = bodyText;
    this.bodyJson = bodyJson;
  }

  private static makeMessage(status: number, bodyText: string, bodyJson: any | null) {
    // Backend canonical shape:
    //   { code, message, extra?, request_id? }
    // Also tolerate:
    //   { error: { code, message, ... }, request_id }   (older)
    //   { detail: ... }                                (FastAPI default)
    if (bodyJson && typeof bodyJson === "object") {
      const errObj =
        bodyJson.error && typeof bodyJson.error === "object"
          ? bodyJson.error
          : bodyJson;

      // Try common places for code/message
      const code =
        errObj.code != null
          ? String(errObj.code)
          : bodyJson.code != null
          ? String(bodyJson.code)
          : null;

      const msg =
        errObj.message != null
          ? String(errObj.message)
          : errObj.detail != null
          ? typeof errObj.detail === "string"
            ? errObj.detail
            : JSON.stringify(errObj.detail)
          : bodyJson.message != null
          ? String(bodyJson.message)
          : null;

      const rid = bodyJson.request_id != null ? String(bodyJson.request_id) : null;

      if (code || msg) {
        const base = `HTTP ${status}: ${code ?? "error"}${msg ? ` — ${msg}` : ""}`;
        return rid ? `${base} (request_id=${rid})` : base;
      }
    }

    return `HTTP ${status}: ${bodyText}`;
  }
}

/**
 * Runtime-configurable API base.
 *
 * Precedence:
 *  1) setApiBaseUrl() called by UI bootstrap (recommended; from /config.json)
 *  2) VITE_API_BASE_URL (dev fallback)
 *  3) "/api" (reverse-proxy default)
 */
let API_BASE: string = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim() || "/api";

/**
 * If your backend is hosted at e.g. http://localhost:8000 (no /api),
 * pass base like "http://localhost:8000" and we will normalize to
 * "http://localhost:8000/api".
 *
 * If you pass "http://localhost:8000/api" we keep it as-is.
 */
export function setApiBaseUrl(base: string | undefined | null): void {
  const s = (base ?? "").trim();
  if (!s) return;

  // Allow relative "/api" too.
  if (s === "/api") {
    API_BASE = "/api";
    return;
  }

  // Remove trailing slash
  const noTrail = s.endsWith("/") ? s.slice(0, -1) : s;

  // If caller already includes /api, use it; else append /api.
  API_BASE = noTrail.endsWith("/api") ? noTrail : `${noTrail}/api`;
}

const API_KEY = import.meta.env.VITE_API_KEY as string | undefined;

// ---- Dev UX: warn loudly if missing ----
if (!API_KEY) {
  // eslint-disable-next-line no-console
  console.warn("VITE_API_KEY is not set. Requests will fail with 401 missing_api_key.");
}

function authHeaders(): HeadersInit {
  // Backend expects X-API-Key per backend/src/llm_server/api/deps.py
  return API_KEY ? { "X-API-Key": API_KEY } : {};
}

async function requestJson<T>(path: string, init: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);

  // Read body once
  const bodyText = await res.text();

  // Parse only a bounded preview (prevents slow JSON.parse on huge HTML error pages)
  const preview = bodyText.slice(0, 4000);
  let bodyJson: any | null = null;

  if (preview.trim()) {
    try {
      bodyJson = JSON.parse(preview);
    } catch {
      bodyJson = null;
    }
  }

  if (!res.ok) {
    throw new ApiError(res.status, preview, bodyJson);
  }

  // If empty response, return {} as T
  if (!bodyText.trim()) return {} as T;

  // Prefer parsed JSON if available; otherwise try parsing full body once (success path should be JSON)
  if (bodyJson !== null) return bodyJson as T;

  try {
    const full = JSON.parse(bodyText);
    return full as T;
  } catch {
    throw new ApiError(res.status, `Non-JSON response: ${preview}`, null);
  }
}

// -----------------------------
// API functions
// -----------------------------
export async function callGenerate(body: GenerateRequestBody): Promise<GenerateResponseBody> {
  return requestJson<GenerateResponseBody>("/v1/generate", {
    method: "POST",
    headers: {
      ...authHeaders(),
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
}

export async function callExtract(body: ExtractRequestBody): Promise<ExtractResponseBody> {
  return requestJson<ExtractResponseBody>("/v1/extract", {
    method: "POST",
    headers: {
      ...authHeaders(),
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
}

export async function listSchemas(): Promise<SchemaIndexItem[]> {
  return requestJson<SchemaIndexItem[]>("/v1/schemas", {
    method: "GET",
    headers: {
      ...authHeaders(),
    },
  });
}

export async function getSchema(schemaId: string): Promise<JsonSchema> {
  return requestJson<JsonSchema>(`/v1/schemas/${encodeURIComponent(schemaId)}`, {
    method: "GET",
    headers: {
      ...authHeaders(),
    },
  });
}

export function getApiBaseUrl(): string {
  return API_BASE;
}

// New: /v1/models
export async function listModels(): Promise<ModelsResponseBody> {
  return requestJson<ModelsResponseBody>("/v1/models", {
    method: "GET",
    headers: {
      ...authHeaders(),
    },
  });
}

// Back-compat: callers can keep using getCapabilities(),
// but it now comes from /v1/models.deployment_capabilities.
export async function getCapabilities(): Promise<CapabilitiesResponseBody> {
  const r = await listModels();
  return r.deployment_capabilities;
}