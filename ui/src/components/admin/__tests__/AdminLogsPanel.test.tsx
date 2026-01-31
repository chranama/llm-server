// ui/src/components/admin/__tests__/AdminStatsPanel.test.tsx
import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

import { AdminLogsPanel } from "../AdminLogsPanel";

const adminListLogs = vi.fn();

vi.mock("../../../lib/api", async () => {
  const actual: any = await vi.importActual("../../../lib/api");
  return {
    ...actual,
    adminListLogs: (params: any) => adminListLogs(params),
  };
});

function makePage(total: number, limit: number, offset: number) {
  return {
    total,
    limit,
    offset,
    items: [
      {
        id: 1,
        created_at: "2026-01-31T00:00:00Z",
        api_key: "test_key_123456789",
        route: "/v1/generate",
        client_host: "127.0.0.1",
        model_id: "m1",
        latency_ms: 12.3,
        prompt_tokens: 10,
        completion_tokens: 20,
        prompt: "hi",
        output: "ok",
      },
    ],
  };
}

describe("AdminLogsPanel", () => {
  beforeEach(() => {
    adminListLogs.mockReset();
  });

  it("loads logs on mount with defaults (limit=50, offset=0)", async () => {
    adminListLogs.mockResolvedValueOnce(makePage(1, 50, 0));
    render(<AdminLogsPanel />);

    await waitFor(() => {
      expect(adminListLogs).toHaveBeenCalledWith({
        model_id: undefined,
        route: undefined,
        api_key: undefined,
        limit: 50,
        offset: 0,
      });
    });

    expect(await screen.findByText(/total:/i)).toBeInTheDocument();
    expect(screen.getByText("/v1/generate")).toBeInTheDocument();
    expect(screen.getByText("m1")).toBeInTheDocument();
  });

  it("applies filters and resets offset to 0", async () => {
    adminListLogs
      .mockResolvedValueOnce(makePage(120, 50, 0)) // initial
      .mockResolvedValueOnce(makePage(1, 50, 0)); // after apply filters

    render(<AdminLogsPanel />);

    await waitFor(() => expect(adminListLogs).toHaveBeenCalledTimes(1));

    fireEvent.change(screen.getByPlaceholderText(/model_id filter/i), { target: { value: "m9" } });
    fireEvent.change(screen.getByPlaceholderText(/route filter/i), { target: { value: "/v1/extract" } });
    fireEvent.change(screen.getByPlaceholderText(/api_key filter/i), { target: { value: "abc" } });

    fireEvent.click(screen.getByRole("button", { name: /apply filters/i }));

    await waitFor(() => {
      expect(adminListLogs).toHaveBeenLastCalledWith({
        model_id: "m9",
        route: "/v1/extract",
        api_key: "abc",
        limit: 50,
        offset: 0,
      });
    });
  });

  it("Next button advances offset by limit when possible", async () => {
    adminListLogs
      .mockResolvedValueOnce(makePage(120, 50, 0)) // initial: canNext true
      .mockResolvedValueOnce(makePage(120, 50, 50)); // after next

    render(<AdminLogsPanel />);

    await waitFor(() => expect(adminListLogs).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole("button", { name: /next/i }));

    await waitFor(() => {
      expect(adminListLogs).toHaveBeenLastCalledWith({
        model_id: undefined,
        route: undefined,
        api_key: undefined,
        limit: 50,
        offset: 50,
      });
    });
  });

  it("Prev button goes back by limit", async () => {
    adminListLogs
      .mockResolvedValueOnce(makePage(120, 50, 0)) // initial
      .mockResolvedValueOnce(makePage(120, 50, 50)) // next
      .mockResolvedValueOnce(makePage(120, 50, 0)); // prev

    render(<AdminLogsPanel />);

    await waitFor(() => expect(adminListLogs).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: /next/i }));
    await waitFor(() => expect(adminListLogs).toHaveBeenCalledTimes(2));

    fireEvent.click(screen.getByRole("button", { name: /prev/i }));
    await waitFor(() => expect(adminListLogs).toHaveBeenCalledTimes(3));

    expect(adminListLogs).toHaveBeenLastCalledWith({
      model_id: undefined,
      route: undefined,
      api_key: undefined,
      limit: 50,
      offset: 0,
    });
  });

  it("renders error when request fails", async () => {
    adminListLogs.mockRejectedValueOnce(new Error("fail"));
    render(<AdminLogsPanel />);
    expect(await screen.findByText(/fail/i)).toBeInTheDocument();
  });
});