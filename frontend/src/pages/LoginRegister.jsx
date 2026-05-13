import { useState } from "react";
import { useNavigate } from "react-router-dom";
import AuthForm from "../components/AuthForm";
import api from "../api";
import { saveToken } from "../auth";


function LoginRegister() {
  const [mode, setMode] = useState("login");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const toggleMode = () => {
    setError("");
    setMode((current) => (current === "login" ? "register" : "login"));
  };

  const handleSubmit = async ({ username, password }) => {
    setError("");
    try {
      if (mode === "register") {
        try {
          await api.post("/auth/register", { username, password });
        } catch (err) {
                  console.log(err);
                  console.log(err.response);
                  console.log(err.response?.data);

                  const detail = err.response?.data?.detail || "Registration failed";

                  console.error("Register error:", detail);
                  setError(detail);
                  return;
        }
      }
      const response = await api.post(
        "/auth/login",
        new URLSearchParams({ username, password }),
        { headers: { "Content-Type": "application/x-www-form-urlencoded" } }
      );
      saveToken(response.data.access_token);
      navigate("/");
    } catch (err) {
      const detail = err.response?.data?.detail || "Login failed";
      console.error("Auth error:", detail, err);
      setError(detail);
    }
  };

  return (
    <main className="flex min-h-screen items-center justify-center bg-slate-950 px-4 py-10 sm:px-6">
      <AuthForm mode={mode} onSubmit={handleSubmit} toggleMode={toggleMode} error={error} />
    </main>
  );
}

export default LoginRegister;
