import { useState, useRef, useEffect } from 'react'
import { Pencil, Check, X } from 'lucide-react'

interface Props {
  value: string | number | null | undefined
  onSave: (val: string) => void
  className?: string
  inputClassName?: string
  placeholder?: string
  isHighlighted?: boolean  // true when field came from OCR with low confidence
}

export default function EditableField({
  value,
  onSave,
  className = '',
  inputClassName = '',
  placeholder = '—',
  isHighlighted = false,
}: Props) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(String(value ?? ''))
  const [edited, setEdited] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (editing) inputRef.current?.focus()
  }, [editing])

  const handleSave = () => {
    const trimmed = draft.trim()
    onSave(trimmed)
    setEdited(true)
    setEditing(false)
  }

  const handleCancel = () => {
    setDraft(String(value ?? ''))
    setEditing(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSave()
    if (e.key === 'Escape') handleCancel()
  }

  if (editing) {
    return (
      <span className="inline-flex items-center gap-1">
        <input
          ref={inputRef}
          value={draft}
          onChange={e => setDraft(e.target.value)}
          onKeyDown={handleKeyDown}
          className={`px-1.5 py-0.5 text-sm rounded border border-brand-400 dark:border-brand-600 bg-white dark:bg-stone-800 text-stone-800 dark:text-stone-200 focus:outline-none focus:ring-1 focus:ring-brand-400 min-w-[80px] ${inputClassName}`}
        />
        <button
          onClick={handleSave}
          className="p-0.5 rounded text-green-500 hover:bg-green-50 dark:hover:bg-green-950/30 transition-colors"
        >
          <Check size={12} />
        </button>
        <button
          onClick={handleCancel}
          className="p-0.5 rounded text-stone-400 hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors"
        >
          <X size={12} />
        </button>
      </span>
    )
  }

  return (
    <span
      className={`group inline-flex items-center gap-1 cursor-pointer ${className}`}
      onClick={() => { setDraft(String(value ?? '')); setEditing(true) }}
      title="Click to edit"
    >
      <span className={`${isHighlighted && !edited ? 'text-amber-600 dark:text-amber-400' : ''}`}>
        {value ?? placeholder}
      </span>

      {/* edited badge */}
      {edited && (
        <span className="inline-flex items-center gap-0.5 text-xs px-1.5 py-0.5 rounded-full bg-brand-50 dark:bg-brand-950/30 border border-brand-200 dark:border-brand-800 text-brand-600 dark:text-brand-400">
          <Pencil size={9} />
          edited
        </span>
      )}

      {/* edit icon — visible on hover */}
      {!edited && (
        <Pencil
          size={11}
          className="opacity-0 group-hover:opacity-60 text-stone-400 transition-opacity"
        />
      )}
    </span>
  )
}
