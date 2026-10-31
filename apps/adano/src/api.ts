export type Category = { id: number; name: string; description: string; count: number }
export type Note = {
  id: number
  text: string
  category_id: number | null
  predicted_category_id: number | null
  confidence: number
  is_confirmed: boolean
  created_at: string
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!res.ok) throw new Error(`${res.status}: ${(await res.text()) || res.statusText}`)
  return res.status === 204 ? (undefined as T) : res.json()
}

export const api = {
  categories: () => call<Category[]>('/categories'),
  createCategory: (name: string, description: string) =>
    call<Category>('/categories', { method: 'POST', body: JSON.stringify({ name, description }) }),
  deleteCategory: (id: number) => call<void>(`/categories/${id}`, { method: 'DELETE' }),
  notes: () => call<Note[]>('/notes'),
  createNote: (text: string) => call<Note>('/notes', { method: 'POST', body: JSON.stringify({ text }) }),
  deleteNote: (id: number) => call<void>(`/notes/${id}`, { method: 'DELETE' }),
  setCategory: (id: number, category_id: number | null) =>
    call<Note>(`/notes/${id}/category`, { method: 'PATCH', body: JSON.stringify({ category_id }) }),
  reclassify: () => call<{ changed: number }>('/notes/reclassify', { method: 'POST' }),
}
