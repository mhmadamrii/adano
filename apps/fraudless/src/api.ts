// Mirrors apps/fraudless-api/fraudless/api/schemas.py (contract: docs/DATA_CONTRACT.md).
export type Issue = { field: string; code: string; message: string }
export type RowResult = {
  row_id: number
  transaction_id: string
  prediction: 'potentially_suspicious' | 'not_flagged'
  risk_score: number
  missing_model_inputs: string[]
}
export type RejectedRow = { row_id: number; errors: Issue[] }
export type Warn = { code: string; message: string; fields?: string[] | null; count?: number | null }
export type PredictResponse = {
  run_id: string | null
  model_version: string
  schema_version: string
  total_rows: number
  scored_rows: number
  rejected_row_count: number
  flagged_rows: number
  threshold: number
  results: RowResult[]
  rejected_rows: RejectedRow[]
  rejected_rows_truncated: boolean
  warnings: Warn[]
  explanation: { model_type: string; features_used: string[]; score_interpretation: string; note: string }
  disclaimer: string
}
export type ModelInfo = {
  model_version: string
  schema_version: string
  model_type: string
  threshold: number
  metrics: { test: { pr_auc: number; precision: number; recall: number; prevalence: number } }
  dataset: { name: string; label_definition: string }
  limitations: string
  supported_schema_fields: { name: string; type: string; requirement: string }[]
}
export class ApiError extends Error {
  constructor(public code: string, message: string, public details: Record<string, unknown>, public status: number) {
    super(message)
  }
}

async function parse<T>(res: Response): Promise<T> {
  if (res.ok) return res.json()
  let body: any = null
  try {
    body = await res.json()
  } catch {}
  const e = body?.error
  throw new ApiError(e?.code ?? 'http_error', e?.message ?? `Request failed (${res.status})`, e?.details ?? {}, res.status)
}

export const api = {
  modelInfo: () => fetch('/api/model-info').then((r) => parse<ModelInfo>(r)),
  predict: (file: File, mapping?: string) => {
    const form = new FormData()
    form.append('file', file)
    if (mapping?.trim()) form.append('mapping', mapping.trim())
    return fetch('/api/predict/file', { method: 'POST', body: form }).then((r) => parse<PredictResponse>(r))
  },
}
