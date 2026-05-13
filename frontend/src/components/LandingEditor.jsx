import { useState } from "react";
import Editor from "@monaco-editor/react";

function LandingEditor({ initialCode, onChange }) {
  const [charLimit] = useState(1800);

  return (
    <div className="space-y-4">
      <div className="rounded-3xl border border-slate-800 bg-slate-900/90 p-4 shadow-xl shadow-slate-950/20">
        <div className="flex items-center justify-between pb-3">
          <div>
            <h2 className="text-xl font-semibold text-slate-100">Python Editor</h2>
            <p className="text-sm text-slate-400">Paste or type Python code to get AI-driven recommendations.</p>
          </div>
          <span className="rounded-full bg-slate-800 px-3 py-1 text-sm text-slate-300">
            Soft limit: {charLimit} chars
          </span>
        </div>
        <Editor
          height="380px"
          defaultLanguage="python"
          defaultValue={initialCode}
          theme="vs-dark"
          options={{ minimap: { enabled: false }, wordWrap: "on", fontSize: 14 }}
          onChange={(value) => onChange(value || "")}
        />
      </div>
      <div className="flex flex-col gap-2 rounded-3xl border border-slate-800 bg-slate-900/90 p-4">
        <div className="flex items-center justify-between text-slate-300">
          <span>Character count: {initialCode.length}</span>
          {initialCode.length > charLimit && (
            <span className="text-amber-300">Large files may reduce recommendation quality.</span>
          )}
        </div>
      </div>
    </div>
  );
}

export default LandingEditor;
