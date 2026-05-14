import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import api from "../api";
import LoadingSpinner from "../components/LoadingSpinner";
import { clearToken } from "../auth";
import LandingEditor from "../components/LandingEditor";

function Results() {
  const { id } = useParams();
  const [submission, setSubmission] = useState(null);
  const [code, setCode] = useState("");
  const [charLimit, setCharLimit] = useState(2000);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await api.get("/submissions/config");
        setCharLimit(response.data.char_limit);
      } catch (err) {
        console.warn("Failed to fetch char limit:", err);
      }
    };
    fetchConfig();
  }, []);

  useEffect(() => {
    const loadSubmission = async () => {
      try {
        const response = await api.get(`/submissions/${id}`);
        setSubmission(response.data);
        setCode(response.data.raw_code);
      } catch (err) {
        if (err.response?.data?.detail === "Could not validate credentials") {
          clearToken();
          navigate("/auth");
          return;
        }
        setError(err.response?.data?.detail || "Unable to load results.");
      } finally {
        setLoading(false);
      }
    };

    loadSubmission();
  }, [id, navigate]);

  const handleCopy = async () => {
    if (!submission) return;
    await navigator.clipboard.writeText(submission.recommendation_text);
  };

  const handleResubmit = async () => {
    if (!code) return;
    setSubmitting(true);
    setError("");

    try {
      const response = await api.post("/submissions/", { raw_code: code });
      navigate(`/results/${response.data.id}`);
    } catch (err) {
      if (err.response?.data?.detail === "Could not validate credentials") {
        clearToken();
        navigate("/auth");
      } else {
        setError(err.response?.data?.detail || "Unable to resubmit code.");
      }
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) {
    return <LoadingSpinner />;
  }

  if (error) {
    return (
      <div className="mx-auto mt-20 max-w-3xl rounded-3xl border border-rose-600 bg-slate-900/90 p-8 text-slate-100">
        <p>{error}</p>
      </div>
    );
  }

  return (
    <main className="mx-auto flex min-h-screen max-w-6xl flex-col gap-8 px-4 py-8 sm:px-6 lg:px-8">
      <div className="flex flex-col gap-3 rounded-3xl border border-slate-800 bg-slate-900/90 p-6 shadow-xl shadow-slate-950/30 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="mt-2 text-3xl font-semibold text-white">ML Recommendations</h1>
        </div>
        <div className="flex gap-3">
          <button className="rounded-2xl border border-slate-800 bg-slate-950 px-4 py-3 text-slate-100 transition hover:bg-slate-800" onClick={() => navigate("/")}>
            Back to editor
          </button>
          <button
            className="rounded-2xl bg-cyan-500 px-4 py-3 text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-60"
            onClick={handleResubmit}
            disabled={submitting || !code}
          >
            {submitting ? "Submitting..." : "Resubmit code"}
          </button>
          <button className="rounded-2xl border border-slate-800 bg-slate-950 px-4 py-3 text-slate-100 transition hover:bg-slate-800" onClick={handleCopy}>
            Copy recommendations
          </button>
        </div>
      </div>

      <section className="grid gap-8 xl:grid-cols-[1.8fr_1fr]">
        <div>
          <LandingEditor initialCode={code} onChange={setCode} charLimit={charLimit} />
          {error && <p className="mt-3 rounded-2xl bg-rose-500/10 px-4 py-3 text-sm text-rose-200">{error}</p>}
        </div>
        <div className="space-y-6 rounded-3xl border border-slate-800 bg-slate-900/90 p-6 shadow-xl shadow-slate-950/20">
          <div>
            <h2 className="text-lg font-semibold text-white">Recommendations</h2>
            <p className="mt-4 whitespace-pre-line rounded-3xl bg-slate-950/80 px-4 py-5 text-sm text-slate-200">{submission?.recommendation_text}</p>
          </div>
        </div>
      </section>
    </main>
  );
}

export default Results;
