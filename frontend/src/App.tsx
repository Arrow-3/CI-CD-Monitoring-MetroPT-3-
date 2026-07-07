import { Button } from "@/components/ui/button"

function App() {
  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center">
      <div className="text-center space-y-4">
        <h1 className="text-3xl font-bold tracking-tight text-slate-900">
          MetroPT-3 · Adaptive Condition Monitoring
        </h1>
        <p className="text-slate-600">
          Toolchain check — if this button is styled, everything works.
        </p>
        <Button
          onClick={() => alert("Vite + React + TS + Tailwind + shadcn/ui: OK")}
        >
          Test everything
        </Button>
      </div>
    </div>
  )
}

export default App