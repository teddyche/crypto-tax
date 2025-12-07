import React, { useEffect, useMemo, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://192.168.1.69:8010";

function formatDate(dateStr) {
  const d = new Date(dateStr);
  if (Number.isNaN(d.getTime())) return dateStr;
  return d.toLocaleString("fr-FR", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function classNames(...classes) {
  return classes.filter(Boolean).join(" ");
}

const TYPE_BADGE_COLORS = {
  BUY: "bg-emerald-500/10 text-emerald-300 border border-emerald-500/30",
  SELL: "bg-red-500/10 text-red-300 border border-red-500/30",
  DEPOSIT: "bg-sky-500/10 text-sky-300 border border-sky-500/30",
  WITHDRAWAL: "bg-orange-500/10 text-orange-300 border border-orange-500/30",
  CONVERT: "bg-violet-500/10 text-violet-300 border border-violet-500/30",
  INCOME: "bg-amber-500/10 text-amber-300 border border-amber-500/30",
  SUBSCRIPTION: "bg-slate-500/10 text-slate-300 border border-slate-600/40",
  OTHER: "bg-slate-500/10 text-slate-300 border border-slate-500/30",
};

function formatSideLabel(side) {
  switch (side) {
    case "BUY":
      return "Achat";
    case "SELL":
      return "Vente";
    case "DEPOSIT":
      return "Dépôt";
    case "WITHDRAWAL":
      return "Retrait";
    case "CONVERT":
      return "Convert";
    case "SUBSCRIPTION":
      return "Subscription (Earn)";
    case "INCOME":
      return "Revenu (Earn)";
    default:
      return "Autre";
  }
}

const TYPE_FILTERS = [
  { value: "BUY", label: "Achat" },
  { value: "SELL", label: "Vente" },
  { value: "DEPOSIT", label: "Dépôt" },
  { value: "WITHDRAWAL", label: "Retrait" },
  { value: "CONVERT", label: "Convert" },
  { value: "INCOME", label: "Revenu" },
  { value: "SUBSCRIPTION", label: "Subscription (Earn)" },
  { value: "OTHER", label: "Autre" },
];

function App() {
  const [transactions, setTransactions] = useState([]);
  const [summary, setSummary] = useState({
    total_transactions: 0,
    total_buy: 0,
    total_sell: 0,
    total_deposit: 0,
    total_withdrawal: 0,
    total_convert: 0,
  });

  const [loading, setLoading] = useState(false);
  const [loadingSummary, setLoadingSummary] = useState(false);

  // Filtres
  const [years, setYears] = useState([]);
  const [assets, setAssets] = useState([]);
  const [selectedYear, setSelectedYear] = useState("all");
  const [selectedAsset, setSelectedAsset] = useState("all");

  // Filtre de type (multi)
  const [selectedTypes, setSelectedTypes] = useState([]);

  // Pagination
  const [pageSize, setPageSize] = useState(100);
  const [page, setPage] = useState(1);

  // Upload
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [uploadSuccess, setUploadSuccess] = useState("");

  // --- Helpers filtres & pagination ---

  const buildQueryString = () => {
    const params = new URLSearchParams();
    params.set("limit", String(pageSize));
    params.set("offset", String((page - 1) * pageSize));

    if (selectedYear !== "all") {
      params.set("year", String(selectedYear));
    }
    if (selectedAsset !== "all") {
      params.set("asset", selectedAsset);
    }
    if (selectedTypes.length > 0) {
      selectedTypes.forEach((t) => params.append("types", t));
    }

    return params.toString();
  };

  const fetchFilters = async () => {
    try {
      const paramsYears = new URLSearchParams();
      const paramsAssets = new URLSearchParams();

      if (selectedAsset !== "all") {
        paramsYears.set("asset", selectedAsset);
      }
      if (selectedYear !== "all") {
        paramsAssets.set("year", String(selectedYear));
      }

      const [yearsRes, assetsRes] = await Promise.all([
        fetch(
          `${API_URL}/years${
            paramsYears.toString() ? "?" + paramsYears.toString() : ""
          }`
        ),
        fetch(
          `${API_URL}/assets${
            paramsAssets.toString() ? "?" + paramsAssets.toString() : ""
          }`
        ),
      ]);

      const yearsJson = await yearsRes.json();
      const assetsJson = await assetsRes.json();

      const newYears = yearsJson.years || [];
      const newAssets = assetsJson.assets || [];

      setYears(newYears);
      setAssets(newAssets);

      // Si l’année / l’actif sélectionné n’existe plus, reset
      if (selectedYear !== "all" && !newYears.includes(Number(selectedYear))) {
        setSelectedYear("all");
      }
      if (
        selectedAsset !== "all" &&
        !newAssets.includes(String(selectedAsset))
      ) {
        setSelectedAsset("all");
      }
    } catch (err) {
      console.error("Erreur chargement filtres", err);
    }
  };

  const fetchSummaryAndTransactions = async () => {
    const qs = buildQueryString();

    try {
      setLoading(true);
      setLoadingSummary(true);

      const [txRes, sumRes] = await Promise.all([
        fetch(`${API_URL}/transactions?${qs}`),
        fetch(`${API_URL}/summary?${qs}`),
      ]);

      const txJson = await txRes.json();
      const sumJson = await sumRes.json();

      setTransactions(txJson || []);
      setSummary(
        sumJson || {
          total_transactions: 0,
          total_buy: 0,
          total_sell: 0,
          total_deposit: 0,
          total_withdrawal: 0,
          total_convert: 0,
        }
      );
    } catch (err) {
      console.error("Erreur fetch summary/transactions", err);
    } finally {
      setLoading(false);
      setLoadingSummary(false);
    }
  };

  const pageCount = useMemo(() => {
    const total = summary.total_transactions || 0;
    if (total === 0) return 1;
    return Math.max(1, Math.ceil(total / pageSize));
  }, [summary.total_transactions, pageSize]);

  // --- useEffects ---

  // Reset page quand les filtres changent
  useEffect(() => {
    setPage(1);
  }, [selectedYear, selectedAsset, selectedTypes, pageSize]);

  // Filtres croisés (années <-> actifs)
  useEffect(() => {
    fetchFilters();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedYear, selectedAsset]);

  // Data + summary
  useEffect(() => {
    fetchSummaryAndTransactions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedYear, selectedAsset, selectedTypes, page, pageSize]);

  // --- Upload CSV Binance ---

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadError("");
    setUploadSuccess("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      setUploading(true);
      const res = await fetch(`${API_URL}/import/binance`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`HTTP ${res.status} - ${text}`);
      }

      const json = await res.json();
      setUploadSuccess(`${json.inserted || 0} lignes importées.`);

      await fetchFilters();
      await fetchSummaryAndTransactions();
    } catch (err) {
      console.error(err);
      setUploadError(err.message || "Erreur import CSV");
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  };

  // --- UI helpers ---

  const totalsCards = useMemo(() => {
    return [
      {
        label: "Transactions totales",
        value: summary.total_transactions,
      },
      {
        label: "Achat (BUY)",
        value: summary.total_buy,
      },
      {
        label: "Vente (SELL)",
        value: summary.total_sell,
      },
      {
        label: "Dépôts",
        value: summary.total_deposit,
      },
      {
        label: "Retraits",
        value: summary.total_withdrawal,
      },
      {
        label: "Conversions (CONVERT)",
        value: summary.total_convert,
      },
    ];
  }, [summary]);

  const toggleType = (value) => {
    setSelectedTypes((prev) =>
      prev.includes(value)
        ? prev.filter((v) => v !== value)
        : [...prev, value]
    );
  };

  const clearTypes = () => setSelectedTypes([]);

  const goPrevPage = () => {
    setPage((p) => Math.max(1, p - 1));
  };

  const goNextPage = () => {
    setPage((p) => Math.min(pageCount, p + 1));
  };

  // --- Render ---

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-100 flex">
      {/* Sidebar */}
      <aside className="w-64 border-r border-slate-800 bg-slate-950/80 backdrop-blur-xl hidden md:flex flex-col">
        <div className="px-6 py-4 flex items-center gap-3 border-b border-slate-800">
          <div className="h-9 w-9 rounded-xl bg-emerald-500/10 border border-emerald-400/40 flex items-center justify-center">
            <span className="text-emerald-300 font-semibold text-lg">₿</span>
          </div>
          <div>
            <div className="text-sm font-semibold tracking-tight">
              CryptoTax
            </div>
            <div className="text-xs text-slate-400">Prototype perso</div>
          </div>
        </div>

        <nav className="mt-4 px-3 space-y-1">
          <button className="w-full flex items-center gap-2 px-3 py-2 text-sm rounded-lg bg-slate-800/80 text-slate-100">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            Tableau de bord
          </button>
          <button className="w-full flex items-center gap-2 px-3 py-2 text-sm rounded-lg text-slate-400 hover:bg-slate-800/40">
            <span className="w-1.5 h-1.5 rounded-full bg-slate-500" />
            Transactions
          </button>
          <button className="w-full flex items-center gap-2 px-3 py-2 text-sm rounded-lg text-slate-400 hover:bg-slate-800/40">
            <span className="w-1.5 h-1.5 rounded-full bg-slate-500" />
            Rapports fiscaux
          </button>
          <button className="w-full flex items-center gap-2 px-3 py-2 text-sm rounded-lg text-slate-400 hover:bg-slate-800/40">
            <span className="w-1.5 h-1.5 rounded-full bg-slate-500" />
            Paramètres
          </button>
        </nav>

        <div className="mt-auto px-4 py-4 border-t border-slate-800 text-xs text-slate-500">
          ENV: <span className="text-emerald-300">DEV</span> • Backend FastAPI
          • React + Vite
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 px-4 md:px-10 py-6">
        {/* Header */}
        <header className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-xl md:text-2xl font-semibold tracking-tight">
              Tax return – Binance
            </h1>
            <p className="text-xs md:text-sm text-slate-400 mt-1">
              Prototype perso type Waltio. Front React + FastAPI sur ton NAS.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <span className="hidden md:inline-flex items-center text-xs px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-300 border border-emerald-500/30">
              ENV: DEV
            </span>
            <span className="inline-flex items-center text-xs px-3 py-1 rounded-full bg-slate-800 border border-slate-700 text-slate-300">
              Année fiscale 2021+
            </span>
          </div>
        </header>

        {/* Filtres haut */}
        <section className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-6">
          <div className="flex flex-col gap-3">
            <div className="flex flex-wrap gap-3">
              {/* Année */}
              <div className="flex flex-col text-xs gap-1">
                <span className="text-slate-400 uppercase tracking-wide">
                  Année fiscale
                </span>
                <select
                  value={selectedYear}
                  onChange={(e) => setSelectedYear(e.target.value)}
                  className="bg-slate-900 border border-slate-700 text-sm rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:ring-1 focus:ring-emerald-400/60"
                >
                  <option value="all">Toutes</option>
                  {years.map((y) => (
                    <option key={y} value={y}>
                      {y}
                    </option>
                  ))}
                </select>
              </div>

              {/* Actif */}
              <div className="flex flex-col text-xs gap-1">
                <span className="text-slate-400 uppercase tracking-wide">
                  Actif (BCH, BTC, USDT…)
                </span>
                <select
                  value={selectedAsset}
                  onChange={(e) => setSelectedAsset(e.target.value)}
                  className="bg-slate-900 border border-slate-700 text-sm rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:ring-1 focus:ring-emerald-400/60 min-w-[150px]"
                >
                  <option value="all">Tous</option>
                  {assets.map((asset) => (
                    <option key={asset} value={asset}>
                      {asset}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Types */}
            <div className="flex flex-col text-xs gap-1">
              <span className="text-slate-400 uppercase tracking-wide">
                TYPES D&apos;OPÉRATION
              </span>

              <div className="flex flex-wrap gap-2 mt-1">
                <button
                  type="button"
                  onClick={clearTypes}
                  className={classNames(
                    "px-3 py-1 rounded-full text-[11px] border transition",
                    selectedTypes.length === 0
                      ? "bg-emerald-500/20 border-emerald-400 text-emerald-100"
                      : "bg-slate-900 border-slate-600 text-slate-300 hover:border-slate-400"
                  )}
                >
                  Tous
                </button>

                {TYPE_FILTERS.map((t) => {
                  const active = selectedTypes.includes(t.value);
                  return (
                    <button
                      key={t.value}
                      type="button"
                      onClick={() => toggleType(t.value)}
                      className={classNames(
                        "px-3 py-1 rounded-full text-[11px] border transition",
                        active
                          ? "bg-slate-100 text-slate-900 border-slate-100"
                          : "bg-slate-900 border-slate-600 text-slate-300 hover:border-slate-400"
                      )}
                    >
                      {t.label}
                    </button>
                  );
                })}
              </div>
            </div>
           </div> {/* <-- ferme le <div className="flex flex-col gap-3"> */}

          {/* Upload CSV */}
          <div className="flex items-start gap-3">
            <label className="relative inline-flex items-center justify-center px-4 py-2 rounded-full bg-emerald-500 hover:bg-emerald-400 text-sm font-medium text-slate-950 shadow-lg shadow-emerald-500/30 cursor-pointer transition">
              {uploading ? "Import en cours..." : "Importer CSV Binance"}
              <input
                type="file"
                accept=".csv"
                className="absolute inset-0 opacity-0 cursor-pointer"
                onChange={handleFileChange}
                disabled={uploading}
              />
            </label>
          </div>
        </section>

        {/* Messages upload */}
        {(uploadError || uploadSuccess) && (
          <div className="mb-4 text-xs">
            {uploadError && (
              <div className="px-3 py-2 rounded-md bg-red-500/10 border border-red-500/40 text-red-200 mb-1">
                {uploadError}
              </div>
            )}
            {uploadSuccess && (
              <div className="px-3 py-2 rounded-md bg-emerald-500/10 border border-emerald-500/40 text-emerald-200">
                {uploadSuccess}
              </div>
            )}
          </div>
        )}

        {/* Cards résumé */}
        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-4 mb-6">
          {totalsCards.map((card) => (
            <div
              key={card.label}
              className="rounded-2xl bg-gradient-to-br from-slate-900 to-slate-950 border border-slate-800 px-4 py-3 shadow-sm shadow-slate-900/60"
            >
              <div className="text-[11px] uppercase tracking-wide text-slate-500 mb-1">
                {card.label}
              </div>
              <div className="text-xl font-semibold tabular-nums text-slate-50">
                {loadingSummary ? "…" : card.value}
              </div>
            </div>
          ))}
        </section>

        {/* Tableau transactions */}
        <section className="rounded-2xl bg-slate-950/70 border border-slate-800 shadow-xl shadow-black/50 overflow-hidden">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between px-4 py-3 border-b border-slate-800 text-xs text-slate-400 gap-2">
            <span>Transactions récentes</span>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1">
                <span>
                  Affichage de{" "}
                  <span className="text-slate-200 font-medium">
                    {transactions.length}
                  </span>{" "}
                  lignes
                </span>
                <span className="hidden sm:inline">
                  • Page{" "}
                  <span className="text-slate-200 font-medium">{page}</span> /{" "}
                  <span className="text-slate-200 font-medium">
                    {pageCount}
                  </span>
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="hidden sm:inline">Lignes par page</span>
                <select
                  value={pageSize}
                  onChange={(e) => setPageSize(Number(e.target.value))}
                  className="bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-xs text-slate-100 focus:outline-none focus:ring-1 focus:ring-emerald-400/60"
                >
                  <option value={50}>50</option>
                  <option value={100}>100</option>
                  <option value={150}>150</option>
                </select>
                <button
                  type="button"
                  onClick={goPrevPage}
                  disabled={page <= 1}
                  className={classNames(
                    "px-2 py-1 rounded-md border text-xs",
                    page <= 1
                      ? "border-slate-700 text-slate-600 cursor-not-allowed"
                      : "border-slate-600 text-slate-200 hover:border-slate-300"
                  )}
                >
                  ←
                </button>
                <button
                  type="button"
                  onClick={goNextPage}
                  disabled={page >= pageCount || summary.total_transactions === 0}
                  className={classNames(
                    "px-2 py-1 rounded-md border text-xs",
                    page >= pageCount || summary.total_transactions === 0
                      ? "border-slate-700 text-slate-600 cursor-not-allowed"
                      : "border-slate-600 text-slate-200 hover:border-slate-300"
                  )}
                >
                  →
                </button>
              </div>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full text-xs">
              <thead className="bg-slate-900/80 border-b border-slate-800">
                <tr className="text-[11px] uppercase tracking-wide text-slate-500">
                  <th className="px-4 py-2 text-left font-medium">Date</th>
                  <th className="px-3 py-2 text-left font-medium">Type</th>
                  <th className="px-3 py-2 text-left font-medium">
                    Pair / Coin
                  </th>
                  <th className="px-3 py-2 text-right font-medium">Quantité</th>
                  <th className="px-3 py-2 text-right font-medium">Prix EUR</th>
                  <th className="px-3 py-2 text-right font-medium">Frais EUR</th>
                  <th className="px-4 py-2 text-left font-medium">Note</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td
                      colSpan={7}
                      className="px-4 py-6 text-center text-slate-500"
                    >
                      Chargement…
                    </td>
                  </tr>
                ) : transactions.length === 0 ? (
                  <tr>
                    <td
                      colSpan={7}
                      className="px-4 py-6 text-center text-slate-500"
                    >
                      Aucune transaction pour ce filtre.
                    </td>
                  </tr>
                ) : (
                  transactions.map((tx) => {
                    const sideUpper = (tx.side || "").toUpperCase();
                    const badgeClass =
                      TYPE_BADGE_COLORS[sideUpper] ||
                      TYPE_BADGE_COLORS.OTHER;

                    const qtyNumber = Number(tx.quantity ?? 0);
                    let qtyColor = "text-slate-200";
                    if (qtyNumber > 0) qtyColor = "text-emerald-400";
                    else if (qtyNumber < 0) qtyColor = "text-red-400";

                    return (
                      <tr
                        key={tx.id}
                        className="border-b border-slate-900/60 hover:bg-slate-900/60 transition-colors"
                      >
                        <td className="px-4 py-2 text-slate-300 whitespace-nowrap">
                          {formatDate(tx.datetime)}
                        </td>
                        <td className="px-3 py-2">
                          <span
                            className={classNames(
                              "inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-medium",
                              badgeClass
                            )}
                          >
                            {formatSideLabel(sideUpper)}
                          </span>
                        </td>
                        <td className="px-3 py-2 text-slate-200">
                          {tx.pair || "—"}
                        </td>
                        <td
                          className={classNames(
                            "px-3 py-2 text-right tabular-nums",
                            qtyColor
                          )}
                        >
                          {tx.quantity?.toLocaleString("fr-FR", {
                            maximumFractionDigits: 8,
                          }) || "0"}
                        </td>
                        <td className="px-3 py-2 text-right text-slate-400 tabular-nums">
                          {tx.price_eur
                            ? tx.price_eur.toLocaleString("fr-FR", {
                                minimumFractionDigits: 2,
                                maximumFractionDigits: 2,
                              }) + " €"
                            : "0,00 €"}
                        </td>
                        <td className="px-3 py-2 text-right text-slate-400 tabular-nums">
                          {tx.fees_eur
                            ? tx.fees_eur.toLocaleString("fr-FR", {
                                minimumFractionDigits: 2,
                                maximumFractionDigits: 2,
                              }) + " €"
                            : "0,00 €"}
                        </td>
                        <td className="px-4 py-2 text-slate-400 max-w-xs truncate">
                          {tx.note || "—"}
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;