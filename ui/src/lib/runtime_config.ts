// ui/src/lib/runtime_config.ts
export type UiRuntimeConfig = {
  api?: {
    base_url?: string;
    timeout_ms?: number;
  };
  features?: {
    schema_discovery?: boolean;
    schema_inspector?: boolean;
    json_diff?: boolean;
    curl_generator?: boolean;
  };
  defaults?: {
    mode?: "extract" | "generate";
    schema_id?: string;
    model_id?: string;
  };
  ui?: {
    max_payload_chars?: number;
    pretty_print_json?: boolean;
  };
};

const FALLBACK: Required<Pick<UiRuntimeConfig, "api">> = {
  api: {
    base_url: (import.meta as any).env?.VITE_API_BASE_URL || "http://127.0.0.1:8000",
    timeout_ms: Number((import.meta as any).env?.VITE_API_TIMEOUT_MS || 60000),
  },
};

export async function loadRuntimeConfig(): Promise<UiRuntimeConfig> {
  try {
    // cache-bust so updates take effect without hard refresh in some setups
    const res = await fetch(`/config.json?ts=${Date.now()}`, { cache: "no-store" });
    if (!res.ok) return FALLBACK;
    const data = (await res.json()) as UiRuntimeConfig;
    return {
      ...FALLBACK,
      ...data,
      api: { ...FALLBACK.api, ...(data.api || {}) },
    };
  } catch {
    return FALLBACK;
  }
}