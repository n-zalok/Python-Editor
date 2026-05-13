import { useState } from "react";

function AuthForm({ onSubmit, mode, toggleMode, error }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  return (
    <div className="mx-auto w-full max-w-md rounded-3xl border border-slate-800 bg-slate-900/90 p-8 shadow-2xl shadow-slate-950/40">
      <h1 className="mb-4 text-3xl font-semibold text-white">{mode === "login" ? "Login" : "Register"}</h1>
      <form
        className="space-y-5"
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit({ username, password });
        }}
      >
        <label className="block text-slate-300">
          <span className="text-sm">Username</span>
          <input
            className="mt-2 w-full rounded-2xl border border-slate-800 bg-slate-950 px-4 py-3 text-slate-100 outline-none transition focus:border-cyan-500"
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            required
          />
        </label>
        <label className="block text-slate-300">
          <span className="text-sm">Password</span>
          <input
            type="password"
            className="mt-2 w-full rounded-2xl border border-slate-800 bg-slate-950 px-4 py-3 text-slate-100 outline-none transition focus:border-cyan-500"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            required
          />
        </label>
        {error && <p className="text-sm text-rose-400">{error}</p>}
        <button className="w-full rounded-2xl bg-cyan-500 px-4 py-3 font-semibold text-slate-950 transition hover:bg-cyan-400">
          {mode === "login" ? "Login" : "Create account"}
        </button>
      </form>
      <button className="mt-4 text-sm text-slate-400 hover:text-slate-100" onClick={toggleMode}>
        {mode === "login" ? "Need an account? Register" : "Already have an account? Login"}
      </button>
    </div>
  );
}

export default AuthForm;
