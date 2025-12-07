// frontend/src/App.jsx
import { useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "http://192.168.1.69:8010";

function formatDate(dateStr) {
  try {
    const d = new Date(dateStr);
    return d.toLocaleString("fr-FR", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return dateStr;
  }
}

function formatNumber(n, decimals = 8) {
  if (n === null || n === undefined) return "-";
  return Number(n).toLocaleString("fr-FR", {
    minimumFractionDigits: 0,
    maximumFractionDigits: decimals,
  });
}

function mapSideToLabel(side) {
  switch (side) {
    case "BUY":
      return "Achat";
    case "SELL":
      return "Vente";
    case "DEPOSIT":
      return "Dépôt";
    case "WITHDRAWAL":
      return "Retrait";
    default:
      return side || "Autre";
  }
}

function mapSideToTagColor(side) {
  switch (side) {
    case "BUY":
      return "#22c55e"; // vert
    case "SELL":
      return "#f97316"; // orange
    case "DEPOSIT":
      return "#0ea5e9"; // bleu
    case "WITHDRAWAL":
      return "#e11d48"; // rose/rouge
    default:
      return "#a3a3a3"; // gris
  }
}

function App() {
  const [transactions, setTransactions] = useState([]);
  const [summary, setSummary] = useState(null);

  const [yearsOptions, setYearsOptions] = useState([]);
  const [assetsOptions, setAssetsOptions] = useState([]);

  const [year, setYear] = useState("");
  const [asset, setAsset] = useState("");
  const [loading, setLoading] = useState(false);

  // Chargement des options (années + actifs) au démarrage
  useEffect(() => {
    async function loadFilters() {
      try {
        const [yearsRes, assetsRes] = await Promise.all([
          fetch(`${API_BASE}/years`),
          fetch(`${API_BASE}/assets`),
        ]);

        const yearsJson = await yearsRes.json();
        const assetsJson = await assetsRes.json();

        setYearsOptions(yearsJson.years || []);
        setAssetsOptions(assetsJson.assets || []);
      } catch (err) {
        console.error("Erreur loadFilters:", err);
      }
    }

    loadFilters();
  }, []);

  // Récupération summary + transactions
  async function refreshData() {
    try {
      setLoading(true);

      const params = new URLSearchParams();
      if (year) params.set("year", year);
      if (asset) params.set("asset", asset);

      const [summaryRes, txRes] = await Promise.all([
        fetch(`${API_BASE}/summary?` + params.toString()),
        fetch(`${API_BASE}/transactions?limit=100&` + params.toString()),
      ]);

      const summaryJson = await summaryRes.json();
      const txJson = await txRes.json();

      setSummary(summaryJson);
      setTransactions(txJson || []);
    } catch (err) {
      console.error("Erreur refreshData:", err);
    } finally {
      setLoading(false);
    }
  }

  // premier chargement
  useEffect(() => {
    refreshData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "radial-gradient(circle at top, #0f172a 0, #020617 55%, #000 100%)",
        color: "#f9fafb",
        display: "flex",
        fontFamily:
          "-apple-system, BlinkMacSystemFont, system-ui, -apple-system, system-ui, sans-serif",
      }}
    >
      {/* SIDEBAR */}
      <aside
        style={{
          width: 230,
          borderRight: "1px solid rgba(148,163,184,0.15)",
          padding: "18px 16px",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          background:
            "linear-gradient(to bottom, rgba(15,23,42,0.98), rgba(15,23,42,0.96), rgba(15,23,42,0.9))",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 24 }}>
            <div
              style={{
                width: 28,
                height: 28,
                borderRadius: 999,
                background: "conic-gradient(from 140deg, #22c55e, #22d3ee, #6366f1, #22c55e)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 16,
                fontWeight: 700,
              }}
            >
              ₿
            </div>
            <div>
              <div style={{ fontSize: 15, fontWeight: 600 }}>CryptoTax</div>
              <div style={{ fontSize: 11, color: "#9ca3af" }}>Prototype perso</div>
            </div>
          </div>

          <nav style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <div
              style={{
                padding: "8px 10px",
                borderRadius: 8,
                background: "rgba(148,163,184,0.12)",
                border: "1px solid rgba(94,234,212,0.4)",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                fontSize: 13,
              }}
            >
              <span>Tableau de bord</span>
              <span
                style={{
                  fontSize: 10,
                  padding: "2px 6px",
                  borderRadius: 999,
                  background: "rgba(16,185,129,0.15)",
                  border: "1px solid rgba(52,211,153,0.6)",
                  color: "#6ee7b7",
                }}
              >
                BINANCE
              </span>
            </div>

            <div
              style={{
                padding: "8px 10px",
                borderRadius: 8,
                fontSize: 13,
                color: "#6b7280",
              }}
            >
              Transactions
            </div>
            <div
              style={{
                padding: "8px 10px",
                borderRadius: 8,
                fontSize: 13,
                color: "#6b7280",
              }}
            >
              Rapports fiscaux
            </div>
            <div
              style={{
                padding: "8px 10px",
                borderRadius: 8,
                fontSize: 13,
                color: "#6b7280",
              }}
            >
              Paramètres
            </div>
          </nav>
        </div>

        <div
          style={{
            marginTop: 24,
            padding: "10px 10px",
            borderRadius: 12,
            background: "rgba(15,23,42,0.9)",
            border: "1px solid rgba(148,163,184,0.35)",
          }}
        >
          <div style={{ fontSize: 11, color: "#9ca3af", marginBottom: 4 }}>
            ENV : <span style={{ color: "#22c55e" }}>DEV</span>
          </div>
          <div style={{ fontSize: 11, color: "#9ca3af" }}>Backend FastAPI sur NAS.</div>
        </div>
      </aside>

      {/* MAIN */}
      <main style={{ flex: 1, padding: "20px 28px 32px 28px", overflowX: "hidden" }}>
        {/* HEADER */}
        <header
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginBottom: 22,
          }}
        >
          <div>
            <h1 style={{ fontSize: 24, fontWeight: 600, marginBottom: 6 }}>
              Tax return – Binance
            </h1>
            <p style={{ fontSize: 13, color: "#9ca3af" }}>
              Prototype perso type Waltio. Front React + FastAPI sur ton NAS.
            </p>
          </div>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
            }}
          >
            <div
              style={{
                fontSize: 11,
                color: "#a3e635",
                padding: "4px 8px",
                borderRadius: 999,
                border: "1px solid rgba(190,242,100,0.6)",
                background: "rgba(26,86,219,0.15)",
              }}
            >
              Année fiscale 2021+
            </div>
          </div>
        </header>

        {/* BARRE DE FILTRES */}
        <section
          style={{
            marginBottom: 18,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 12,
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "flex-end",
              gap: 16,
              flexWrap: "wrap",
            }}
          >
            {/* Année fiscale */}
            <div>
              <label
                style={{
                  display: "block",
                  fontSize: 11,
                  textTransform: "uppercase",
                  letterSpacing: 0.04,
                  color: "#9ca3af",
                  marginBottom: 4,
                }}
              >
                Année fiscale
              </label>
              <select
                value={year}
                onChange={(e) => setYear(e.target.value)}
                style={{
                  background: "rgba(15,23,42,0.9)",
                  borderRadius: 999,
                  border: "1px solid rgba(55,65,81,0.9)",
                  padding: "6px 12px",
                  fontSize: 13,
                  minWidth: 110,
                  color: "#e5e7eb",
                }}
              >
                <option value="">Toutes</option>
                {yearsOptions.map((y) => (
                  <option key={y} value={y}>
                    {y}
                  </option>
                ))}
              </select>
            </div>

            {/* Actif */}
            <div>
              <label
                style={{
                  display: "block",
                  fontSize: 11,
                  textTransform: "uppercase",
                  letterSpacing: 0.04,
                  color: "#9ca3af",
                  marginBottom: 4,
                }}
              >
                Actif (BCH, BTC, USDT…)
              </label>
              <select
                value={asset}
                onChange={(e) => setAsset(e.target.value)}
                style={{
                  background: "rgba(15,23,42,0.9)",
                  borderRadius: 999,
                  border: "1px solid rgba(55,65,81,0.9)",
                  padding: "6px 12px",
                  fontSize: 13,
                  minWidth: 140,
                  color: "#e5e7eb",
                }}
              >
                <option value="">Tous</option>
                {assetsOptions.map((a) => (
                  <option key={a} value={a}>
                    {a}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Bouton Mettre à jour */}
          <button
            onClick={refreshData}
            style={{
              padding: "8px 22px",
              borderRadius: 999,
              border: "none",
              background:
                "radial-gradient(circle at 10% 0%, #bbf7d0 0, #22c55e 30%, #22c55e 70%, #16a34a 100%)",
              color: "#022c22",
              fontSize: 13,
              fontWeight: 600,
              boxShadow: "0 0 20px rgba(34,197,94,0.4)",
              cursor: "pointer",
              whiteSpace: "nowrap",
            }}
          >
            {loading ? "Chargement..." : "Mettre à jour"}
          </button>
        </section>

        {/* SUMMARY CARDS */}
        <section
          style={{
            display: "flex",
            gap: 16,
            flexWrap: "wrap",
            marginBottom: 26,
          }}
        >
          {[
            {
              label: "Transactions totales",
              value: summary?.total_transactions ?? 0,
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
          ].map((card, idx) => (
            <div
              key={idx}
              style={{
                flex: "0 0 200px",
                maxWidth: 220,
                background:
                  "linear-gradient(to bottom right, rgba(15,23,42,0.9), rgba(15,23,42,0.95))",
                borderRadius: 16,
                padding: "10px 14px",
                border: "1px solid rgba(31,41,55,0.9)",
                boxShadow: "0 18px 40px rgba(15,23,42,0.85)",
              }}
            >
              <div
                style={{
                  fontSize: 11,
                  textTransform: "uppercase",
                  letterSpacing: 0.08,
                  color: "#9ca3af",
                  marginBottom: 4,
                }}
              >
                {card.label}
              </div>
              <div style={{ fontSize: 20, fontWeight: 600 }}>
                {formatNumber(card.value, 0)}
              </div>
            </div>
          ))}
        </section>

        {/* TABLEAU TRANSACTIONS */}
        <section
          style={{
            background: "rgba(15,23,42,0.92)",
            borderRadius: 18,
            border: "1px solid rgba(31,41,55,0.95)",
            boxShadow: "0 20px 45px rgba(15,23,42,0.95)",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              padding: "10px 16px",
              borderBottom: "1px solid rgba(31,41,55,0.9)",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              fontSize: 12,
              color: "#9ca3af",
            }}
          >
            <div>Transactions récentes</div>
            <div>Affichage des 100 dernières lignes</div>
          </div>

          <div style={{ overflowX: "auto" }}>
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                fontSize: 12,
              }}
            >
              <thead
                style={{
                  background:
                    "linear-gradient(to right, rgba(15,23,42,0.98), rgba(15,23,42,0.96))",
                }}
              >
                <tr>
                  <th style={thStyle}>Date</th>
                  <th style={thStyle}>Type</th>
                  <th style={thStyle}>Pair / Coin</th>
                  <th style={thStyle}>Quantité</th>
                  <th style={thStyle}>Prix EUR</th>
                  <th style={thStyle}>Frais EUR</th>
                  <th style={thStyle}>Note</th>
                </tr>
              </thead>
              <tbody>
                {transactions.length === 0 && (
                  <tr>
                    <td
                      colSpan={7}
                      style={{
                        padding: "16px 18px",
                        textAlign: "center",
                        color: "#6b7280",
                      }}
                    >
                      Aucune transaction à afficher pour ces filtres.
                    </td>
                  </tr>
                )}

                {transactions.map((tx) => {
                  const sideLabel = mapSideToLabel(tx.side);
                  const tagColor = mapSideToTagColor(tx.side);

                  return (
                    <tr
                      key={tx.id}
                      style={{
                        borderTop: "1px solid rgba(31,41,55,0.85)",
                        background:
                          "radial-gradient(circle at top left, rgba(15,23,42,0.96), rgba(15,23,42,0.94))",
                      }}
                    >
                      <td style={tdStyle}>{formatDate(tx.datetime)}</td>

                      {/* Type avec badge couleur */}
                      <td style={tdStyle}>
                        <span
                          style={{
                            fontSize: 11,
                            padding: "3px 8px",
                            borderRadius: 999,
                            border: `1px solid ${tagColor}`,
                            color: tagColor,
                            background: "rgba(15,23,42,0.9)",
                          }}
                        >
                          {sideLabel}
                        </span>
                      </td>

                      <td style={tdStyle}>{tx.pair}</td>
                      <td style={{ ...tdStyle, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                        {formatNumber(tx.quantity)}
                      </td>
                      <td style={{ ...tdStyle, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                        {formatNumber(tx.price_eur, 2)}
                      </td>
                      <td style={{ ...tdStyle, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                        {formatNumber(tx.fees_eur, 4)}
                      </td>
                      <td style={{ ...tdStyle, maxWidth: 260 }}>
                        <span
                          style={{
                            display: "inline-block",
                            whiteSpace: "nowrap",
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            maxWidth: 250,
                            color: "#9ca3af",
                          }}
                          title={tx.note || ""}
                        >
                          {tx.note || "—"}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  );
}

// Styles de cellules pour le tableau
const thStyle = {
  textAlign: "left",
  padding: "8px 14px",
  fontWeight: 500,
  color: "#9ca3af",
  borderBottom: "1px solid rgba(31,41,55,0.9)",
  whiteSpace: "nowrap",
};

const tdStyle = {
  padding: "8px 14px",
  color: "#e5e7eb",
  whiteSpace: "nowrap",
};

export default App;