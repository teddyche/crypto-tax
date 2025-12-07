import { useEffect, useState } from "react";

const API_BASE = `http://${window.location.hostname}:8010`;

function App() {
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchTransactions = async () => {
      try {
        const res = await fetch(`${API_BASE}/transactions`);
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const data = await res.json();
        setTransactions(data);
      } catch (err) {
        console.error(err);
        setError("Impossible de charger les transactions.");
      } finally {
        setLoading(false);
      }
    };

    fetchTransactions();
  }, []);

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", padding: "1.5rem" }}>
      <h1 style={{ fontSize: "1.8rem", marginBottom: "0.5rem" }}>
        NOOOOOO
      </h1>
      <p style={{ marginBottom: "1.5rem", color: "#555" }}>
        YES
      </p>

      {loading && <p>Chargement des transactions…</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}

      {!loading && !error && (
        <>
          {transactions.length === 0 ? (
            <p>Aucune transaction pour le moment.</p>
          ) : (
            <table
              style={{
                borderCollapse: "collapse",
                width: "100%",
                maxWidth: "1000px",
              }}
            >
              <thead>
                <tr>
                  <th style={thStyle}>Date</th>
                  <th style={thStyle}>Exchange</th>
                  <th style={thStyle}>Pair</th>
                  <th style={thStyle}>Side</th>
                  <th style={thStyle}>Quantity</th>
                  <th style={thStyle}>Prix (EUR)</th>
                  <th style={thStyle}>Frais (EUR)</th>
                  <th style={thStyle}>Note</th>
                </tr>
              </thead>
              <tbody>
                {transactions.map((tx) => (
                  <tr key={tx.id}>
                    <td style={tdStyle}>
                      {new Date(tx.datetime).toLocaleString()}
                    </td>
                    <td style={tdStyle}>{tx.exchange}</td>
                    <td style={tdStyle}>{tx.pair}</td>
                    <td style={tdStyle}>
                      <span
                        style={{
                          padding: "2px 6px",
                          borderRadius: "4px",
                          fontSize: "0.8rem",
                          backgroundColor:
                            tx.side === "BUY" ? "#d1fae5" : "#fee2e2",
                          color: tx.side === "BUY" ? "#065f46" : "#b91c1c",
                        }}
                      >
                        {tx.side}
                      </span>
                    </td>
                    <td style={tdStyle}>{tx.quantity}</td>
                    <td style={tdStyle}>{tx.price_eur.toFixed(2)}</td>
                    <td style={tdStyle}>{tx.fees_eur.toFixed(2)}</td>
                    <td style={tdStyle}>{tx.note || "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      )}
    </div>
  );
}

const thStyle = {
  borderBottom: "1px solid #ddd",
  textAlign: "left",
  padding: "8px",
  backgroundColor: "#f9fafb",
};

const tdStyle = {
  borderBottom: "1px solid #eee",
  padding: "8px",
  fontSize: "0.9rem",
};

export default App;
