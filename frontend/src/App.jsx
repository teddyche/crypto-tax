import { useEffect, useState } from "react";

function App() {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("http://192.168.1.69:8010/summary")
      .then((res) => res.json())
      .then((data) => {
        setSummary(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Erreur summary", err);
        setLoading(false);
      });
  }, []);

  return (
    <div style={{ padding: "24px", fontFamily: "system-ui" }}>
      <h1>CryptoTax – Dev</h1>
      <p>Prototype perso type Waltio (Binance pour l’instant).</p>

      {loading && <p>Chargement du résumé…</p>}

      {summary && (
        <div style={{ display: "flex", gap: "16px", margin: "16px 0" }}>
          <Card label="Transactions totales" value={summary.total_transactions} />
          <Card label="Achat (BUY)" value={summary.total_buy} />
          <Card label="Vente (SELL)" value={summary.total_sell} />
          <Card label="Dépôts" value={summary.total_deposit} />
          <Card label="Retraits" value={summary.total_withdrawal} />
        </div>
      )}

      {/* ici tu gardes ton tableau de transactions existant */}
    </div>
  );
}

function Card({ label, value }) {
  return (
    <div
      style={{
        padding: "12px 16px",
        borderRadius: "8px",
        border: "1px solid #ddd",
        minWidth: "150px",
      }}
    >
      <div style={{ fontSize: "12px", color: "#666" }}>{label}</div>
      <div style={{ fontSize: "20px", fontWeight: 600 }}>{value}</div>
    </div>
  );
}

export default App;