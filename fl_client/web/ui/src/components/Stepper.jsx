import { Check } from 'lucide-react';

export default function Stepper({ steps, currentStep }) {
  return (
    <div className="flex items-center w-full">
      {steps.map((step, i) => {
        const isActive = i === currentStep;
        const isDone = i < currentStep;
        const isLast = i === steps.length - 1;

        return (
          <div key={i} className="flex items-center flex-1 last:flex-none">
            {/* Step circle + info */}
            <div className="flex items-center gap-2.5">
              <div
                className={`flex items-center justify-center w-8 h-8 rounded-full text-xs font-semibold shrink-0 transition-colors ${
                  isDone
                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                    : isActive
                    ? 'bg-accent/20 text-accent border border-accent/30'
                    : 'bg-surface-3 text-slate-500 border border-border'
                }`}
              >
                {isDone ? <Check size={14} /> : i + 1}
              </div>
              <div className="hidden sm:block min-w-0">
                <p className={`text-xs font-medium ${isActive ? 'text-slate-200' : isDone ? 'text-slate-300' : 'text-slate-500'}`}>
                  {step.label}
                </p>
                {step.desc && (
                  <p className="text-2xs text-slate-600 truncate">{step.desc}</p>
                )}
              </div>
            </div>

            {/* Connector */}
            {!isLast && (
              <div className="flex-1 mx-3 h-px">
                <div
                  className={`h-full transition-colors ${
                    isDone ? 'bg-emerald-500/40' : 'bg-border'
                  }`}
                />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
