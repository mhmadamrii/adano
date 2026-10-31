import { createFileRoute } from '@tanstack/react-router'
import { useEffect, useMemo, useState } from 'react'
import { api, ApiError, type ModelInfo, type PredictResponse } from '../api'

export const Route = createFileRoute('/')({ component: Home })

const MAPPING_PLACEHOLDER = '{"amount": "Amt", "payment_method": "Method", "transaction_timestamp": "When"}'

function Home() {
  const [info, setInfo] = useState<ModelInfo | null>(null)
  const [infoError, setInfoError] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [mapping, setMapping] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<ApiError | Error | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [flaggedOnly, setFlaggedOnly] = useState(false)

  useEffect(() => {
    api.modelInfo().then(setInfo).catch((e) => setInfoError(String(e.message ?? e)))
  }, [])

  async function run(e: React.FormEvent) {
    e.preventDefault()
    if (!file) return
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      setResult(await api.predict(file, mapping))
    } catch (err) {
      setError(err as Error)
    } finally {
      setBusy(false)
    }
  }

  const rows = useMemo(() => {
    const r = [...(result?.results ?? [])].sort((a, b) => b.risk_score - a.risk_score)
    return flaggedOnly ? r.filter((x) => x.prediction === 'potentially_suspicious') : r
  }, [result, flaggedOnly])

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold">FraudLess</h1>
        <p className="text-sm text-slate-600">Upload a transaction CSV and review which rows look potentially suspicious.</p>
        <ModelBadge info={info} error={infoError} />
      </header>

      <form onSubmit={run} className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
        <label className="block text-sm font-medium" htmlFor="file">
          Transactions file (.csv, UTF-8, comma-separated)
        </label>
        <input
          id="file"
          data-testid="file-input"
          type="file"
          accept=".csv,text/csv"
          className="mt-2 block w-full text-sm file:mr-3 file:rounded file:border-0 file:bg-slate-900 file:px-3 file:py-1.5 file:text-white"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />

        <details className="mt-3 text-sm">
          <summary className="cursor-pointer text-slate-700">Column names differ from the schema? Map them</summary>
          <p className="mt-2 text-slate-600">
            JSON of canonical field to your column header. Required: <code>amount</code>. Fields:{' '}
            {info?.supported_schema_fields.filter((f) => f.requirement !== 'training_only').map((f) => f.name).join(', ') ?? '…'}
          </p>
          <textarea
            data-testid="mapping-input"
            rows={3}
            value={mapping}
            onChange={(e) => setMapping(e.target.value)}
            placeholder={MAPPING_PLACEHOLDER}
            className="mt-2 w-full rounded border border-slate-300 p-2 font-mono text-xs"
          />
        </details>

        <button
          data-testid="run"
          disabled={!file || busy}
          className="mt-4 rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
        >
          {busy ? 'Scoring…' : 'Run audit'}
        </button>
      </form>

      {error && <ErrorBox error={error} />}
      {result && (
        <Results result={result} rows={rows} flaggedOnly={flaggedOnly} setFlaggedOnly={setFlaggedOnly} />
      )}
    </div>
  )
}

function ModelBadge({ info, error }: { info: ModelInfo | null; error: string }) {
  if (error)
    return (
      <p data-testid="api-down" className="mt-3 rounded bg-red-50 p-2 text-sm text-red-800">
        Cannot reach the scoring API ({error}). Start it with <code>pnpm dev:fraudless-api</code>.
      </p>
    )
  if (!info) return null
  const m = info.metrics.test
  return (
    <p data-testid="model-badge" className="mt-3 rounded bg-amber-50 p-2 text-xs text-amber-900">
      Model <b>{info.model_version}</b> ({info.model_type}) · trained on simulated data · test PR-AUC {m.pr_auc.toFixed(2)} (random{' '}
      {m.prevalence.toFixed(4)}), precision {(m.precision * 100).toFixed(0)}%, recall {(m.recall * 100).toFixed(0)}% · proof of concept,
      not a fraud decision system.
    </p>
  )
}

function ErrorBox({ error }: { error: ApiError | Error }) {
  const api = error instanceof ApiError ? error : null
  const rejected = (api?.details.rejected_rows as { row_id: number }[] | undefined) ?? []
  return (
    <div data-testid="error" className="mt-4 rounded border border-red-200 bg-red-50 p-3 text-sm text-red-900">
      <p className="font-medium">
        {api ? `${api.code}: ` : ''}
        {error.message}
      </p>
      {api && Object.keys(api.details).length > 0 && (
        <pre className="mt-2 overflow-x-auto whitespace-pre-wrap text-xs">{JSON.stringify(api.details, null, 2)}</pre>
      )}
      {rejected.length === 0 && api?.code === 'missing_required_columns' && (
        <p className="mt-2">Use the column mapping above to point a canonical field at one of your columns.</p>
      )}
    </div>
  )
}

function Results(p: {
  result: PredictResponse
  rows: PredictResponse['results']
  flaggedOnly: boolean
  setFlaggedOnly: (v: boolean) => void
}) {
  const { result: r } = p
  return (
    <section className="mt-6 space-y-6" data-testid="results">
      <div className="rounded border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900" data-testid="disclaimer">
        {r.disclaimer}
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Tile label="Rows in file" value={r.total_rows} testid="t-total" />
        <Tile label="Scored" value={r.scored_rows} testid="t-scored" />
        <Tile label="Rejected" value={r.rejected_row_count} testid="t-rejected" tone={r.rejected_row_count ? 'warn' : undefined} />
        <Tile label="Potentially suspicious" value={r.flagged_rows} testid="t-flagged" tone={r.flagged_rows ? 'alert' : undefined} />
      </div>
      <p className="text-xs text-slate-600">
        Flag threshold {r.threshold.toFixed(4)}. Risk score is a model-estimated probability under the training prevalence (about 0.13%),
        not a certainty. Run {r.run_id ?? 'not recorded'}.
      </p>

      {r.warnings.length > 0 && (
        <div data-testid="warnings">
          <h2 className="mb-2 text-sm font-semibold">Data quality warnings</h2>
          <ul className="list-disc space-y-1 pl-5 text-sm text-slate-700">
            {r.warnings.map((w, i) => (
              <li key={i}>
                <code className="text-xs">{w.code}</code> {w.message}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div>
        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-sm font-semibold">Scored rows</h2>
          <div className="flex items-center gap-4 text-sm">
            <label className="flex items-center gap-1">
              <input type="checkbox" data-testid="flagged-only" checked={p.flaggedOnly} onChange={(e) => p.setFlaggedOnly(e.target.checked)} />
              Flagged only
            </label>
            <button data-testid="download" className="rounded border border-slate-300 px-2 py-1" onClick={() => download(r)}>
              Download CSV
            </button>
          </div>
        </div>
        <div className="overflow-x-auto rounded border border-slate-200 bg-white">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-100 text-xs uppercase text-slate-600">
              <tr>
                <th className="px-3 py-2">Row</th>
                <th className="px-3 py-2">Transaction</th>
                <th className="px-3 py-2">Result</th>
                <th className="px-3 py-2 text-right">Risk score</th>
                <th className="px-3 py-2">Unknown inputs</th>
              </tr>
            </thead>
            <tbody>
              {p.rows.map((x) => (
                <tr key={x.row_id} data-testid="result-row" data-prediction={x.prediction} className="border-t border-slate-100">
                  <td className="px-3 py-2">{x.row_id}</td>
                  <td className="px-3 py-2 font-mono text-xs">{x.transaction_id}</td>
                  <td className="px-3 py-2">
                    {x.prediction === 'potentially_suspicious' ? (
                      <span className="rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-800">potentially suspicious</span>
                    ) : (
                      <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-700">not flagged</span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums">{x.risk_score.toFixed(4)}</td>
                  <td className="px-3 py-2 text-xs text-slate-600">{x.missing_model_inputs.join(', ') || '-'}</td>
                </tr>
              ))}
              {p.rows.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-3 py-4 text-center text-slate-500">
                    No rows to show.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {r.rejected_rows.length > 0 && (
        <div data-testid="rejected">
          <h2 className="mb-2 text-sm font-semibold">
            Rejected rows ({r.rejected_row_count}){r.rejected_rows_truncated && ', first ' + r.rejected_rows.length + ' shown'}
          </h2>
          <ul className="space-y-1 text-sm">
            {r.rejected_rows.map((x) => (
              <li key={x.row_id} className="rounded border border-red-100 bg-white p-2">
                <b>Row {x.row_id}</b>: {x.errors.map((e) => `${e.field} (${e.message})`).join('; ')}
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  )
}

function Tile({ label, value, testid, tone }: { label: string; value: number; testid: string; tone?: 'warn' | 'alert' }) {
  const color = tone === 'alert' ? 'text-red-700' : tone === 'warn' ? 'text-amber-700' : 'text-slate-900'
  return (
    <div className="rounded border border-slate-200 bg-white p-3">
      <div className="text-xs text-slate-600">{label}</div>
      <div data-testid={testid} className={`text-2xl font-semibold tabular-nums ${color}`}>
        {value}
      </div>
    </div>
  )
}

function csvCell(v: string | number): string {
  const s = String(v)
  // quote, and neutralize spreadsheet formula injection from user-supplied ids
  const safe = /^[=+\-@\t\r]/.test(s) ? `'${s}` : s
  return /[",\n]/.test(safe) ? `"${safe.replace(/"/g, '""')}"` : safe
}

function download(r: PredictResponse) {
  const lines = ['row_id,transaction_id,prediction,risk_score,missing_model_inputs']
  for (const x of r.results)
    lines.push([x.row_id, x.transaction_id, x.prediction, x.risk_score, x.missing_model_inputs.join('|')].map(csvCell).join(','))
  const url = URL.createObjectURL(new Blob([lines.join('\n') + '\n'], { type: 'text/csv' }))
  const a = Object.assign(document.createElement('a'), { href: url, download: `fraudless-${r.run_id ?? 'results'}.csv` })
  a.click()
  URL.revokeObjectURL(url)
}
