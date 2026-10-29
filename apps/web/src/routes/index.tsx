import { createFileRoute } from '@tanstack/react-router'
import { useCallback, useEffect, useState } from 'react'
import { api, type Category, type Note } from '../api'

export const Route = createFileRoute('/')({ component: Home })

const UNCAT = 'uncategorized'

function Home() {
  const [cats, setCats] = useState<Category[]>([])
  const [notes, setNotes] = useState<Note[]>([])
  const [filter, setFilter] = useState<number | typeof UNCAT | null>(null)
  const [text, setText] = useState('')
  const [name, setName] = useState('')
  const [desc, setDesc] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const refresh = useCallback(async () => {
    const [c, n] = await Promise.all([api.categories(), api.notes()])
    setCats(c)
    setNotes(n)
  }, [])

  const run = useCallback(
    async (fn: () => Promise<unknown>) => {
      setError('')
      setBusy(true)
      try {
        await fn()
        await refresh()
      } catch (e) {
        setError(String(e))
      } finally {
        setBusy(false)
      }
    },
    [refresh],
  )

  useEffect(() => {
    run(async () => {})
  }, [run])

  const catName = (id: number | null) => cats.find((c) => c.id === id)?.name ?? 'Uncategorized'
  const shown = notes.filter((n) =>
    filter === null ? true : filter === UNCAT ? n.category_id === null : n.category_id === filter,
  )

  return (
    <div style={{ display: 'flex', minHeight: '100vh' }}>
      <aside style={{ width: 280, padding: 16, borderRight: '1px solid #ddd', background: '#fff' }}>
        <h2 style={{ marginTop: 0 }}>Categories</h2>
        <div style={{ display: 'grid', gap: 4 }}>
          <button data-testid="filter-all" onClick={() => setFilter(null)} style={navStyle(filter === null)}>
            All ({notes.length})
          </button>
          {cats.map((c) => (
            <div key={c.id} style={{ display: 'flex', gap: 4 }}>
              <button
                data-testid={`filter-${c.name}`}
                onClick={() => setFilter(c.id)}
                title={c.description}
                style={{ ...navStyle(filter === c.id), flex: 1 }}
              >
                {c.name} ({c.count})
              </button>
              <button aria-label={`delete ${c.name}`} onClick={() => run(() => api.deleteCategory(c.id))}>
                ×
              </button>
            </div>
          ))}
          <button
            data-testid="filter-uncategorized"
            onClick={() => setFilter(UNCAT)}
            style={navStyle(filter === UNCAT)}
          >
            Uncategorized ({notes.filter((n) => n.category_id === null).length})
          </button>
        </div>

        <form
          style={{ display: 'grid', gap: 6, marginTop: 20 }}
          onSubmit={(e) => {
            e.preventDefault()
            if (!name.trim()) return
            run(async () => {
              await api.createCategory(name, desc)
              setName('')
              setDesc('')
            })
          }}
        >
          <strong>New category</strong>
          <input data-testid="cat-name" placeholder="Name" value={name} onChange={(e) => setName(e.target.value)} />
          <textarea
            data-testid="cat-desc"
            placeholder="Description (helps cold start)"
            rows={3}
            value={desc}
            onChange={(e) => setDesc(e.target.value)}
          />
          <button data-testid="cat-add" disabled={busy}>
            Add category
          </button>
        </form>
        <button style={{ marginTop: 12 }} disabled={busy} onClick={() => run(api.reclassify)}>
          Reclassify unconfirmed
        </button>
      </aside>

      <main style={{ flex: 1, padding: 24, maxWidth: 760 }}>
        <h1 style={{ marginTop: 0 }}>Notes</h1>
        <form
          style={{ display: 'flex', gap: 8 }}
          onSubmit={(e) => {
            e.preventDefault()
            if (!text.trim()) return
            run(async () => {
              await api.createNote(text)
              setText('')
            })
          }}
        >
          <input
            data-testid="note-input"
            style={{ flex: 1, padding: 8 }}
            placeholder="Write a note…"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <button data-testid="note-add" disabled={busy}>
            Add
          </button>
        </form>
        {error && (
          <p data-testid="error" style={{ color: 'crimson' }}>
            {error}
          </p>
        )}

        <ul style={{ listStyle: 'none', padding: 0, display: 'grid', gap: 8, marginTop: 16 }}>
          {shown.map((n) => (
            <li
              key={n.id}
              data-testid="note"
              style={{ background: '#fff', border: '1px solid #ddd', borderRadius: 8, padding: 12 }}
            >
              <div>{n.text}</div>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 8, fontSize: 13 }}>
                <span data-testid="note-category" style={chip(n.category_id !== null)}>
                  {catName(n.category_id)}
                </span>
                <span style={{ color: '#666' }}>
                  {n.is_confirmed ? 'confirmed' : `${Math.round(n.confidence * 100)}%`}
                </span>
                {!n.is_confirmed && n.category_id !== null && (
                  <button data-testid="note-confirm" onClick={() => run(() => api.setCategory(n.id, n.category_id))}>
                    ✓ accept
                  </button>
                )}
                <select
                  data-testid="note-select"
                  value={n.category_id ?? ''}
                  onChange={(e) => run(() => api.setCategory(n.id, e.target.value ? Number(e.target.value) : null))}
                >
                  <option value="">Uncategorized</option>
                  {cats.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name}
                    </option>
                  ))}
                </select>
                <button
                  data-testid="note-delete"
                  aria-label="remove note"
                  style={{ marginLeft: 'auto' }}
                  onClick={() => run(() => api.deleteNote(n.id))}
                >
                  Remove
                </button>
              </div>
            </li>
          ))}
          {shown.length === 0 && <li style={{ color: '#888' }}>No notes.</li>}
        </ul>
      </main>
    </div>
  )
}

const navStyle = (active: boolean): React.CSSProperties => ({
  textAlign: 'left',
  padding: '6px 10px',
  border: '1px solid #ddd',
  borderRadius: 6,
  background: active ? '#111' : '#fff',
  color: active ? '#fff' : '#111',
  cursor: 'pointer',
})

const chip = (assigned: boolean): React.CSSProperties => ({
  padding: '2px 8px',
  borderRadius: 999,
  background: assigned ? '#e0f2fe' : '#fee2e2',
})
