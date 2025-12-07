import { useEffect, useState } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://192.168.1.69:8010";

function formatDate(iso) {
  if (!iso) return "-";
  const d = new Date(iso);
  return d.toLocaleString("fr-FR", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatNumber(n, decimals = 2) {
  if (n === null || n === undefined) return "-";
  return n.toLocaleString("fr-FR", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function SideBadge({ side }) {
  const s = (side || "").toUpperCase();

  let label = s;
  let className = "side-badge side-badge--other";

  if (s === "BUY") {
    label = "Achat";
    className = "side-badge side-badge--buy";
  } else if (s === "SELL") {
    label = "Vente";
    className = "side-badge side-badge--sell";
  } else if (s === "DEPOSIT") {
    label = "Dépôt";
    className = "side-badge side-badge--deposit";
  } else if (s === "WITHDRAWAL") {
    label = "Retrait";
    className = "side-badge side-badge--withdraw";
  }

  return <span className={className}>{label}</span>;
}

function App() {
  const [summary, setSummary] = useState(null);
  const [transactions, setTransactions] = useState([]);
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [loadingTx, setLoadingTx] = useState(false);
  const [error, setError] = useState(null);

  const [year, setYear] = useState("");
  const [asset, setAsset] = useState("");

  async function fetchSummary(params = {}) {
    try {
      setLoadingSummary(true);
      setError(null);

      const url = new URL(`${API_URL}/summary`);
      if (params.year) url.searchParams.set("year", params.year);
      if (params.asset) url.searchParams.set("asset", params.asset);

      const res = await fetch(url);
      if (!res.ok) throw new Error(`Erreur API summary: ${res.status}`);
      const data = await res.json();
      setSummary(data);
    } catch (e) {
      console.error(e);
      setError("Impossible de charger le résumé.");
    } finally {
      setLoadingSummary(false);
    }
  }

  async function fetchTransactions(params = {}) {
    try {
      setLoadingTx(true);
      setError(null);

      const url = new URL(`${API_URL}/transactions`);
      url.searchParams.set("limit", 100);
      url.searchParams.set("offset", 0);
      // On pourrait filtrer côté backend plus tard (year/asset)

      const res = await fetch(url);
      if (!res.ok) throw new Error(`Erreur API transactions: ${res.status}`);
      const data = await res.json();
      setTransactions(data);
    } catch (e) {
      console.error(e);
      setError("Impossible de charger les transactions.");
    } finally {
      setLoadingTx(false);
    }
  }

  function handleApplyFilters() {
    const filters = {
      year: year ? Number(year) : undefined,
      asset: asset || undefined,
    };
    fetchSummary(filters);
    // plus tard : filtrage transactions côté API
  }

  useEffect(() => {
    fetchSummary({});
    fetchTransactions({});
  }, []);

  const totalTx = summary?.total_transactions ?? 0;
  const cards = [
    {
      label: "Transactions totales",
      value: totalTx,
    },
    {
      label: "Achat (BUY)",
      value: summary?.total_buy ?? 0,
    },
    {
      label: "Vente (SELL)",
      value: summary?.total_sell ?? 0,
    },
    {
      label: "Dépôts",
      value: summary?.total_deposit ?? 0,
    },
    {
      label: "Retraits",
      value: summary?.total_withdrawal ?? 0,
    },
  ];

  return (
    <div className="app-shell">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo-circle">€</div>
          <div>
            <div className="logo-title">CryptoTax</div>
            <div className="logo-sub">Prototype perso</div>
          </div>
        </div>

        <nav className="sidebar-nav">
          <div className="nav-item nav-item--active">
            <span className="nav-dot" />
            Tableau de bord
          </div>
          <div className="nav-item">Transactions</div>
          <div className="nav-item">Rapports fiscaux</div>
          <div className="nav-item">Paramètres</div>
        </nav>

        <div className="sidebar-footer">
          <div className="plan-badge">FREE PLAN • 2025</div>
          <div className="plan-text">
            1100 transactions sur 50 (on s’en fout, c’est nous le boss 😄)
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="main">
        <header className="main-header">
          <div>
            <h1 className="main-title">Tax return – Binance</h1>
            <p className="main-subtitle">
              Prototype perso type Waltio. Front React + FastAPI sur ton NAS.
            </p>
          </div>
          <div className="header-right">
            <span className="badge badge--env">ENV: DEV</span>
            <span className="badge badge--year">Année fiscale 2021+</span>
          </div>
        </header>

        {/* Filters */}
        <section className="filters-row">
          <div className="filter-group">
            <label>Année fiscale</label>
            <select
              value={year}
              onChange={(e) => setYear(e.target.value)}
              className="filter-input"
            >
              <option value="">Toutes</option>
              <option value="2020">2020</option>
              <option value="2021">2021</option>
              <option value="2022">2022</option>
              <option value="2023">2023</option>
              <option value="2024">2024</option>
              <option value="2025">2025</option>
            </select>
          </div>

          <div className="filter-group">
            <label>Actif (ex: BCH, BTC, USDT…)</label>
            <input
              className="filter-input"
              placeholder="BCH"
              value={asset}
              onChange={(e) => setAsset(e.target.value)}
            />
          </div>

          <button className="btn-primary" onClick={handleApplyFilters}>
            Mettre à jour
          </button>
        </section>

        {/* Summary cards */}
        <section className="cards-row">
          {cards.map((card) => (
            <div key={card.label} className="stat-card">
              <div className="stat-label">{card.label}</div>
              <div className="stat-value">
                {loadingSummary ? "…" : card.value.toLocaleString("fr-FR")}
              </div>
            </div>
          ))}
        </section>

        {/* Error */}
        {error && <div className="error-banner">{error}</div>}

        {/* Transactions table */}
        <section className="table-section">
          <div className="table-header">
            <h2>Transactions récentes</h2>
            <div className="table-header-meta">
              {loadingTx ? (
                <span>Chargement…</span>
              ) : (
                <span>
                  Affichage des{" "}
                  <strong>{transactions.length.toLocaleString("fr-FR")}</strong>{" "}
                  dernières lignes
                </span>
              )}
            </div>
          </div>

          <div className="table-wrapper">
            <table className="tx-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Type</th>
                  <th>Pair / Coin</th>
                  <th className="th-right">Quantité</th>
                  <th className="th-right">Prix EUR</th>
                  <th className="th-right">Frais EUR</th>
                  <th>Note</th>
                </tr>
              </thead>
              <tbody>
                {transactions.length === 0 && !loadingTx && (
                  <tr>
                    <td colSpan={7} className="empty-row">
                      Aucune transaction à afficher.
                    </td>
                  </tr>
                )}

                {transactions.map((tx) => (
                  <tr key={tx.id}>
                    <td>{formatDate(tx.datetime)}</td>
                    <td>
                      <SideBadge side={tx.side} />
                    </td>
                    <td>{tx.pair}</td>
                    <td className="td-right">
                      {formatNumber(tx.quantity, 8)}
                    </td>
                    <td className="td-right">
                      {formatNumber(tx.price_eur, 2)}
                    </td>
                    <td className="td-right">
                      {formatNumber(tx.fees_eur, 4)}
                    </td>
                    <td className="td-note">{tx.note || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;