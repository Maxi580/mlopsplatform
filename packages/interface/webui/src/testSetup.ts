import "@testing-library/jest-dom/vitest";

// jsdom has no canvas to draw on, so charts stay empty in tests; their current values still show.
vi.mock("uplot", () => ({
  default: class {
    setData() {}
    setSize() {}
    destroy() {}
  },
}));
globalThis.ResizeObserver ??= class {
  observe() {}
  unobserve() {}
  disconnect() {}
};
