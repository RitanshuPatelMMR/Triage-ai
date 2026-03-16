interface Props { text: string }

export default function NoteQualityHint({ text }: Props) {
  const len = text.trim().length

  const getQuality = () => {
    if (len === 0) return null
    if (len < 30) return { label: 'Too short', color: 'text-red-500', hint: 'Add more clinical detail' }
    if (len < 80) return { label: 'Minimal', color: 'text-amber-500', hint: 'Consider adding more context' }
    if (len < 200) return { label: 'Good', color: 'text-green-500', hint: 'Enough detail for analysis' }
    return { label: 'Detailed', color: 'text-green-600', hint: 'Excellent — full analysis possible' }
  }

  const quality = getQuality()
  if (!quality) return null

  return (
    <div className="flex items-center justify-between text-xs">
      <span className="text-stone-400 dark:text-stone-500">{len} characters</span>
      <span className={`font-medium ${quality.color}`}>
        {quality.label} — {quality.hint}
      </span>
    </div>
  )
}