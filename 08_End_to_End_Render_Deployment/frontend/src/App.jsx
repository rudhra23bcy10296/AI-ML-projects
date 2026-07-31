import React, { useState, useRef, useEffect } from "react";

const currency = (n) =>
  n.toLocaleString("en-US", { maximumFractionDigits: 0 });

function useCountUp(target, duration = 700) {
  const [display, setDisplay] = useState(0);
  const fromRef = useRef(0);
  const rafRef = useRef(null);

  useEffect(() => {
    const from = fromRef.current;
    const to = target;
    const start = performance.now();

    cancelAnimationFrame(rafRef.current);

    function tick(now) {
      const elapsed = now - start;
      const t = Math.min(1, elapsed / duration);
      // ease-out cubic
      const eased = 1 - Math.pow(1 - t, 3);
      const value = from + (to - from) * eased;
      setDisplay(value);
      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        fromRef.current = to;
      }
    }

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [target, duration]);

  return display;
}

function Stepper({ label, value, onChange, min, max, step = 1 }) {
  return (
    <div className="field">
      <label>{label}</label>
      <div className="stepper">
        <button
          type="button"
          aria-label={`Decrease ${label}`}
          onClick={() => onChange(Math.max(min, +(value - step).toFixed(2)))}
        >
          −
        </button>
        <span className="stepper-value">{value}</span>
        <button
          type="button"
          aria-label={`Increase ${label}`}
          onClick={() => onChange(Math.min(max, +(value + step).toFixed(2)))}
        >
          +
        </button>
      </div>
    </div>
  );
}

export default function App() {
  const [bedrooms, setBedrooms] = useState(3);
  const [bathrooms, setBathrooms] = useState(2);
  const [sqftLiving, setSqftLiving] = useState(1800);
  const [yearBuilt, setYearBuilt] = useState(1998);

  const [price, setPrice] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const displayPrice = useCountUp(price ?? 0);

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bedrooms,
          bathrooms,
          sqft_living: sqftLiving,
          yr_built: yearBuilt,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Something went wrong.");
      setPrice(data.price);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const low = price ? price * 0.9 : 0;
  const high = price ? price * 1.1 : 0;

  return (
    <div className="page">
      <div className="ambient-glow" aria-hidden="true" />

      <header className="topbar">
        <span className="wordmark">Estimate</span>
        <span className="eyebrow">AI Home Valuation</span>
      </header>

      <main className="hero">
        <h1>
          Know what your home
          <br />
          is <em>really</em> worth.
        </h1>
        <p className="subtitle">
          A linear regression model trained on 4,500+ real King County home
          sales — enter your details for an instant estimate.
        </p>

        <div className="panels">
          <form className="panel form-panel" onSubmit={handleSubmit}>
            <h2>Property details</h2>

            <Stepper
              label="Bedrooms"
              value={bedrooms}
              onChange={setBedrooms}
              min={0}
              max={12}
            />
            <Stepper
              label="Bathrooms"
              value={bathrooms}
              onChange={setBathrooms}
              min={0}
              max={10}
              step={0.25}
            />

            <div className="field">
              <label htmlFor="area">
                Floor area <span className="unit">sq ft</span>
              </label>
              <input
                id="area"
                type="number"
                min={100}
                value={sqftLiving}
                onChange={(e) => setSqftLiving(Number(e.target.value))}
                required
              />
            </div>

            <div className="field">
              <label htmlFor="year">Year built</label>
              <input
                id="year"
                type="number"
                min={1800}
                max={2026}
                value={yearBuilt}
                onChange={(e) => setYearBuilt(Number(e.target.value))}
                required
              />
            </div>

            <button className="submit" type="submit" disabled={loading}>
              {loading ? "Calculating…" : "Get estimate"}
            </button>

            {error && <p className="error">{error}</p>}
          </form>

          <div className="panel result-panel">
            <h2>Estimated value</h2>

            {price === null ? (
              <p className="placeholder">
                Fill in the details and press{" "}
                <strong>Get estimate</strong> to see a valuation here.
              </p>
            ) : (
              <>
                <div className="price">
                  <span className="currency">$</span>
                  {currency(Math.round(displayPrice))}
                </div>

                <div className="range">
                  <div className="range-track">
                    <div className="range-fill" />
                  </div>
                  <div className="range-labels">
                    <span>${currency(Math.round(low))}</span>
                    <span className="range-mid">likely range</span>
                    <span>${currency(Math.round(high))}</span>
                  </div>
                </div>
              </>
            )}

            <dl className="specs">
              <div>
                <dt>Bedrooms</dt>
                <dd>{bedrooms}</dd>
              </div>
              <div>
                <dt>Bathrooms</dt>
                <dd>{bathrooms}</dd>
              </div>
              <div>
                <dt>Area</dt>
                <dd>{currency(sqftLiving)} sq ft</dd>
              </div>
              <div>
                <dt>Built</dt>
                <dd>{yearBuilt}</dd>
              </div>
            </dl>
          </div>
        </div>
      </main>

      <footer className="footnote">
        <span>Model: Linear Regression (scikit-learn)</span>
        <span>AI/ML course project</span>
      </footer>
    </div>
  );
}
