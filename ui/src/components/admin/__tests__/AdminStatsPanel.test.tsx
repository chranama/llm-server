// ui/src/components/admin/__tests__/AdminStatsPanel.test.tsx
import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

import { AdminStatsPanel } from "../AdminStatsPanel";

const adminGetStats = vi.fn();

vi.mock("../../../lib/api", async () => {
  const actual: any = await vi.importActual("../../../lib/api");
  return {
    ...actual,
    adminGetStats: (windowDays: number) => adminGetStats(windowDays),
  };
});

describe("AdminStatsPanel", () => {
  beforeEach(() => {
    adminGetStats.mockReset();
  });

  it("loads stats on mount using default 30 days", async () => {
    adminGetStats.mockResolvedValueOnce({
      window_days: 30,
      since: "2026-01-01T00:00:00Z",
      total_requests: 2,
      total_prompt_tokens: 10,
      total_completion_tokens: 20,
      avg_latency_ms: 12.3,
      per_model: [
        {
          model_id: "m1",
          total_requests: 2,
          total_prompt_tokens: 10,
          total_completion_tokens: 20,
          avg_latency_ms: 12.3,
        },
      ],
    });

    render(<AdminStatsPanel />);

    await waitFor(() => expect(adminGetStats).toHaveBeenCalledWith(30));

    expect(screen.getByText(/total requests:/i)).toHaveTextContent("2");
    expect(screen.getByText(/prompt tokens:/i)).toHaveTextContent("10");
    expect(screen.getByText(/completion tokens:/i)).toHaveTextContent("20");
    expect(screen.getByText(/avg latency:/i)).toHaveTextContent("12.3");

    expect(screen.getByText("m1")).toBeInTheDocument();
  });

  it("reloads when windowDays changes", async () => {
    adminGetStats
      .mockResolvedValueOnce({
        window_days: 30,
        since: "2026-01-01T00:00:00Z",
        total_requests: 0,
        total_prompt_tokens: 0,
        total_completion_tokens: 0,
        avg_latency_ms: null,
        per_model: [],
      })
      .mockResolvedValueOnce({
        window_days: 7,
        since: "2026-01-24T00:00:00Z",
        total_requests: 1,
        total_prompt_tokens: 3,
        total_completion_tokens: 4,
        avg_latency_ms: 50,
        per_model: [
          {
            model_id: "m2",
            total_requests: 1,
            total_prompt_tokens: 3,
            total_completion_tokens: 4,
            avg_latency_ms: 50,
          },
        ],
      });

    render(<AdminStatsPanel />);

    await waitFor(() => expect(adminGetStats).toHaveBeenCalledWith(30));

    const input = screen.getByLabelText(/window \(days\)/i);
    fireEvent.change(input, { target: { value: "7" } });

    await waitFor(() => expect(adminGetStats).toHaveBeenCalledWith(7));
    expect(await screen.findByText("m2")).toBeInTheDocument();
  });

  it("renders error when API call fails", async () => {
    adminGetStats.mockRejectedValueOnce(new Error("nope"));
    render(<AdminStatsPanel />);
    expect(await screen.findByText(/nope/i)).toBeInTheDocument();
  });

  it("clicking Refresh calls load again", async () => {
    adminGetStats
      .mockResolvedValueOnce({
        window_days: 30,
        since: "2026-01-01T00:00:00Z",
        total_requests: 0,
        total_prompt_tokens: 0,
        total_completion_tokens: 0,
        avg_latency_ms: null,
        per_model: [],
      })
      .mockResolvedValueOnce({
        window_days: 30,
        since: "2026-01-01T00:00:00Z",
        total_requests: 5,
        total_prompt_tokens: 1,
        total_completion_tokens: 2,
        avg_latency_ms: null,
        per_model: [],
      });

    render(<AdminStatsPanel />);
    await waitFor(() => expect(adminGetStats).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole("button", { name: /refresh/i }));
    await waitFor(() => expect(adminGetStats).toHaveBeenCalledTimes(2));
    expect(await screen.findByText(/total requests:/i)).toHaveTextContent("5");
  });
});