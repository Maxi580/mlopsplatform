import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { CHART_HEIGHT_PX, CHART_Y_AXIS_WIDTH_PX } from "../config";

// `color` names a CSS colour variable of the theme, e.g. `--accent`.
export type ChartSeries = { label: string; color: string };

/** A uPlot line chart over time with each series' current value above it. */
export default function LineChart({
  title,
  series,
  times,
  values,
  format,
}: {
  title: string;
  series: ChartSeries[];
  // Seconds since the epoch, and per series one value (or null for a gap) per time.
  times: number[];
  values: (number | null)[][];
  format: (value: number | null) => string;
}) {
  const container = useRef<HTMLDivElement>(null);
  const chart = useRef<uPlot>(null);

  // 1. The chart, made once for its series and as wide as its card.
  useEffect(() => {
    const element = container.current;
    if (!element) return;
    // The canvas can't read CSS variables, so they are resolved for the current theme.
    const theme = getComputedStyle(element);
    const color = (variable: string) => theme.getPropertyValue(variable).trim();
    const axis = { stroke: color("--text-muted"), grid: { stroke: color("--border") } };
    const options: uPlot.Options = {
      width: element.clientWidth,
      height: CHART_HEIGHT_PX,
      legend: { show: false },
      scales: { y: { range: (_, min, max) => [Math.min(0, min), Math.max(max, 1e-9)] } },
      axes: [
        axis,
        {
          ...axis,
          size: CHART_Y_AXIS_WIDTH_PX,
          values: (_, ticks) => ticks.map((tick) => format(tick)),
        },
      ],
      series: [
        {},
        ...series.map((line) => ({ label: line.label, stroke: color(line.color), width: 2 })),
      ],
    };
    chart.current = new uPlot(options, [[], ...series.map(() => [])], element);
    const resize = new ResizeObserver(() =>
      chart.current?.setSize({ width: element.clientWidth, height: CHART_HEIGHT_PX }),
    );
    resize.observe(element);
    return () => {
      resize.disconnect();
      chart.current?.destroy();
    };
  }, [series, format]);

  // 2. Its data, each time a reading arrives.
  useEffect(() => {
    chart.current?.setData([times, ...values] as uPlot.AlignedData);
  }, [times, values]);

  return (
    <figure className="card chart">
      <figcaption>
        <h3>{title}</h3>
        <dl className="chart-current">
          {series.map(({ label, color }, index) => (
            <div key={label}>
              <dt>
                <span className="swatch" style={{ background: `var(${color})` }} /> {label}
              </dt>
              <dd>{format(values[index].at(-1) ?? null)}</dd>
            </div>
          ))}
        </dl>
      </figcaption>
      <div ref={container} />
    </figure>
  );
}
