import { AgentState } from "@/types/dashboard";

export interface ResearchFindingsProps {
  state: AgentState;
  setState: (state: AgentState) => void;
}

export function ResearchFindings({ state, setState }: ResearchFindingsProps) {
  return (
    <div className="bg-white/10 backdrop-blur-md p-6 rounded-2xl shadow-xl max-w-3xl w-full">
      <h2 className="text-2xl font-bold text-white mb-1 text-center">
        Research Findings
      </h2>
      <p className="text-slate-400 text-center text-sm italic mb-4">
        Live results from MiroThinker deep-research agent
      </p>
      <hr className="border-white/10 my-4" />
      <div className="flex flex-col gap-3">
        {state.findings?.map((finding, index) => (
          <div
            key={index}
            className="bg-white/5 p-4 rounded-xl text-slate-200 relative group hover:bg-white/10 transition-all"
          >
            <p className="pr-8 text-sm leading-relaxed">{finding}</p>
            <button
              onClick={() =>
                setState({
                  ...state,
                  findings: state.findings?.filter((_, i) => i !== index),
                })
              }
              className="absolute right-3 top-3 opacity-0 group-hover:opacity-100 transition-opacity
                bg-red-500/80 hover:bg-red-600 text-white rounded-full h-5 w-5 flex items-center justify-center text-xs"
            >
              x
            </button>
          </div>
        ))}
      </div>
      {(!state.findings || state.findings.length === 0) && (
        <p className="text-center text-slate-500 italic my-8">
          No findings yet. Ask MiroThinker a research question to get started.
        </p>
      )}

      {state.sources && state.sources.length > 0 && (
        <>
          <hr className="border-white/10 my-4" />
          <h3 className="text-sm font-semibold text-slate-400 mb-2">
            Sources ({state.sources.length})
          </h3>
          <div className="flex flex-wrap gap-2">
            {state.sources.map((source, index) => (
              <span
                key={index}
                className="bg-indigo-500/20 text-indigo-300 text-xs px-2 py-1 rounded-md"
              >
                {source}
              </span>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
