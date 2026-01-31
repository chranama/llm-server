// ui/src/components/admin/__tests__/AdminPage.test.tsx
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

import { AdminPage } from "../AdminPage";

// Mock child panels to keep this test focused on AdminPage behavior.
vi.mock("../AdminStatsPanel", () => ({
  AdminStatsPanel: () => <div data-testid="admin-stats-panel">stats</div>,
}));
vi.mock("../AdminLogsPanel", () => ({
  AdminLogsPanel: () => <div data-testid="admin-logs-panel">logs</div>,
}));

const adminLoadModel = vi.fn();

vi.mock("../../../lib/api", async () => {
  const actual: any = await vi.importActual("../../../lib/api");
  return {
    ...actual,
    adminLoadModel: (body: any) => adminLoadModel(body),
  };
});

describe("AdminPage", () => {
  beforeEach(() => {
    adminLoadModel.mockReset();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders header + panels", () => {
    render(<AdminPage />);
    expect(screen.getByText("Admin")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /load model/i })).toBeInTheDocument();
    expect(screen.getByTestId("admin-stats-panel")).toBeInTheDocument();
    expect(screen.getByTestId("admin-logs-panel")).toBeInTheDocument();
  });

  it("shows success message for already_loaded and clears after timeout", async () => {
    adminLoadModel.mockResolvedValueOnce({
      ok: true,
      already_loaded: true,
      default_model: "m1",
      models: ["m1"],
    });

    render(<AdminPage />);

    fireEvent.click(screen.getByRole("button", { name: /load model/i }));

    expect(await screen.findByText("Model already loaded")).toBeInTheDocument();

    // should clear after ~2.5s
    vi.advanceTimersByTime(2600);
    await waitFor(() => {
      expect(screen.queryByText("Model already loaded")).not.toBeInTheDocument();
    });
  });

  it("shows success message for load triggered", async () => {
    adminLoadModel.mockResolvedValueOnce({
      ok: true,
      already_loaded: false,
      default_model: "m1",
      models: ["m1"],
    });

    render(<AdminPage />);
    fireEvent.click(screen.getByRole("button", { name: /load model/i }));

    expect(await screen.findByText("Model load triggered")).toBeInTheDocument();
  });

  it("shows error text when adminLoadModel throws", async () => {
    adminLoadModel.mockRejectedValueOnce(new Error("boom"));

    render(<AdminPage />);
    fireEvent.click(screen.getByRole("button", { name: /load model/i }));

    expect(await screen.findByText(/boom/i)).toBeInTheDocument();
  });
});