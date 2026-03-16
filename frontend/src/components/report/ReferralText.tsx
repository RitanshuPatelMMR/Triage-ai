import { Copy, Check } from 'lucide-react'
import { useState } from 'react'
import toast from 'react-hot-toast'

interface Props { text?: string }

export default function ReferralText({ text }: Props) {
  const [copied, setCopied] = useState(false)

  if (!text) return null

  const handleCopy = async () => {
    await navigator.clipboard.writeText(text)
    setCopied(true)
    toast.success('Copied to clipboard')
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="animate-fade-in">
      <div className="flex items-center justify-between mb-2">
        <div className="text-xs font-medium text-stone-400 dark:text-stone-500 uppercase tracking-wider">
          Referral text
        </div>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-lg border border-stone-200 dark:border-stone-700 text-stone-500 dark:text-stone-400 hover:bg-stone-50 dark:hover:bg-stone-800 transition-colors"
        >
          {copied ? <Check size={12} className="text-green-500" /> : <Copy size={12} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <div className="p-3 rounded-lg bg-stone-50 dark:bg-stone-800 border border-stone-200 dark:border-stone-700">
        <p className="text-sm text-stone-700 dark:text-stone-300 leading-relaxed">{text}</p>
      </div>
    </div>
  )
}