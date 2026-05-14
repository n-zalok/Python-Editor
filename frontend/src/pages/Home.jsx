import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import LandingEditor from "../components/LandingEditor";
import LoadingSpinner from "../components/LoadingSpinner";
import api from "../api";
import { isAuthenticated, clearToken } from "../auth";

function Home() {
  const [code, setCode] = useState("");
  const [charLimit, setCharLimit] = useState(2000);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await api.get("/submissions/config");
        setCharLimit(response.data.char_limit);
      } catch (err) {
        // Keep default 2000 if fetch fails
        console.warn("Failed to fetch char limit:", err);
      }
    };
    fetchConfig();
  }, []);

  const handleSubmit = async () => {
    if (!isAuthenticated()) {
      return navigate("/auth");
    }

    setLoading(true);
    setError("");
    try {
      const response = await api.post("/submissions/", { raw_code: code });
      navigate(`/results/${response.data.id}`);
    } catch (err) {
      if (err.response?.data?.detail === "Could not validate credentials") {
        clearToken();
        navigate("/auth");
      } else {
        setError(err.response?.data?.detail || "Submission failed. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="mx-auto flex min-h-screen max-w-6xl flex-col gap-8 px-4 py-8 sm:px-6 lg:px-8">
      <header className="flex flex-col gap-4 rounded-3xl border border-slate-800 bg-slate-900/80 p-6 shadow-xl shadow-slate-950/30 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-sm uppercase tracking-[0.4em] text-cyan-300/80">Python Editor</p>
          <h1 className="mt-3 text-4xl font-semibold text-white sm:text-5xl">ML-powered code review for Python.</h1>
          <p className="mt-4 max-w-2xl text-slate-300">Paste code, get fast recommendations, and keep your workflow moving with a modern minimal interface.</p>
        </div>
        <div className="flex items-center gap-3">
          <button
            className="rounded-2xl bg-slate-800 px-5 py-3 text-sm font-semibold text-slate-100 transition hover:bg-slate-700"
            onClick={() => navigate("/auth")}
          >
            {isAuthenticated() ? "Switch account" : "Login / Register"}
          </button>
        </div>
      </header>

      <section className="grid gap-8 xl:grid-cols-[1.8fr_1fr]">
        <div>
          <LandingEditor initialCode={code} onChange={setCode} charLimit={charLimit} />
          {error && <p className="mt-3 rounded-2xl bg-rose-500/10 px-4 py-3 text-sm text-rose-200">{error}</p>}
        </div>
        <aside className="rounded-3xl border border-slate-800 bg-slate-900/90 p-6 shadow-xl shadow-slate-950/20">
          <div className="space-y-5">
            <button
              onClick={handleSubmit}
              disabled={loading || !code}
              className="w-full rounded-3xl bg-cyan-500 px-5 py-4 text-base font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? "Analyzing..." : "Get Recommendations"}
            </button>
            <div className="rounded-3xl border border-slate-800 bg-slate-950/80 p-5 text-sm text-slate-300">
              <p className="font-semibold text-slate-100">Disclaimer</p>
              <p className="mt-2">The backend stores submissions for future model training to improve performance.</p>
            </div>
          </div>
        </aside>
      </section>

      {loading && <LoadingSpinner />}
    </main>
  );
}

export default Home;
