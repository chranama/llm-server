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
  cache?: boolean;
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

export type JsonSchema = Record<string, any>;

// -----------------------------
// Types: /v1/models
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

export interface CapabilitiesResponseBody {
  generate: boolean;
  extract: boolean;
}

// -----------------------------
// Types: admin endpoints
// -----------------------------
export interface AdminModelStats {
  model_id: string;
  total_requests: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  avg_latency_ms: number | null;
}

export interface AdminStatsResponse {
  window_days: number;
  since: string; // ISO string
  total_requests: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  avg_latency_ms: number | null;
  per_model: AdminModelStats[];
}

export interface AdminLogEntry {
  id: number;
  created_at: string;
  api_key?: string | null;
  route: string;
  client_host?: string | null;
  model_id: string;
  latency_ms?: number | null;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  prompt: string;
  output?: string | null;
}

export interface AdminLogsPage {
  total: number;
  limit: number;
  offset: number;
  items: AdminLogEntry[];
}

export interface AdminLoadModelRequest {
  model_id?: string | null;
}

export interface AdminLoadModelResponse {
  ok: boolean;
  already_loaded: boolean;
  default_model: string;
  models: string[];
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
    if (bodyJson && typeof bodyJson === "object") {
      const errObj =
        bodyJson.error && typeof bodyJson.error === "object"
          ? bodyJson.error
          : bodyJson;

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
 */
let API_BASE: string = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim() || "/api";

export function setApiBaseUrl(base: string | undefined | null): void {
  const s = (base ?? "").trim();
  if (!s) return;

  if (s === "/api") {
    API_BASE = "/api";
    return;
  }

  const noTrail = s.endsWith("/") ? s.slice(0, -1) : s;
  API_BASE = noTrail.endsWith("/api") ? noTrail : `${noTrail}/api`;
}

const API_KEY = import.meta.env.VITE_API_KEY as string | undefined;

if (!API_KEY) {
  // eslint-disable-next-line no-console
  console.warn("VITE_API_KEY is not set. Requests will fail with 401 missing_api_key.");
}

function authHeaders(): HeadersInit {
  return API_KEY ? { "X-API-Key": API_KEY } : {};
}

async function requestJson<T>(path: string, init: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  const bodyText = await res.text();

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

  if (!bodyText.trim()) return {} as T;
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
    headers: { ...authHeaders() },
  });
}

export async function getSchema(schemaId: string): Promise<JsonSchema> {
  return requestJson<JsonSchema>(`/v1/schemas/${encodeURIComponent(schemaId)}`, {
    method: "GET",
    headers: { ...authHeaders() },
  });
}

export function getApiBaseUrl(): string {
  return API_BASE;
}

export async function listModels(): Promise<ModelsResponseBody> {
  return requestJson<ModelsResponseBody>("/v1/models", {
    method: "GET",
    headers: { ...authHeaders() },
  });
}

export async function getCapabilities(): Promise<CapabilitiesResponseBody> {
  const r = await listModels();
  return r.deployment_capabilities;
}

// -----------------------------
// Admin API functions
// -----------------------------
export async function adminGetStats(windowDays: number = 30): Promise<AdminStatsResponse> {
  const qs = new URLSearchParams({ window_days: String(windowDays) });
  return requestJson<AdminStatsResponse>(`/v1/admin/stats?${qs.toString()}`, {
    method: "GET",
    headers: { ...authHeaders() },
  });
}

export async function adminListLogs(params?: {
  model_id?: string;
  api_key?: string;
  route?: string;
  from_ts?: string;
  to_ts?: string;
  limit?: number;
  offset?: number;
}): Promise<AdminLogsPage> {
  const qs = new URLSearchParams();

  if (params?.model_id) qs.set("model_id", params.model_id);
  if (params?.api_key) qs.set("api_key", params.api_key);
  if (params?.route) qs.set("route", params.route);
  if (params?.from_ts) qs.set("from_ts", params.from_ts);
  if (params?.to_ts) qs.set("to_ts", params.to_ts);

  qs.set("limit", String(params?.limit ?? 50));
  qs.set("offset", String(params?.offset ?? 0));

  return requestJson<AdminLogsPage>(`/v1/admin/logs?${qs.toString()}`, {
    method: "GET",
    headers: { ...authHeaders() },
  });
}

export async function adminLoadModel(body: AdminLoadModelRequest): Promise<AdminLoadModelResponse> {
  return requestJson<AdminLoadModelResponse>("/v1/admin/models/load", {
    method: "POST",
    headers: {
      ...authHeaders(),
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body ?? {}),
  });
}