import { AgentStep, AnalysisStatus } from '../../types'
import { CheckCircle, Circle, Loader } from 'lucide-react'

interface Props {
  steps: AgentStep[]
  status: AnalysisStatus
}

const NODE_LABELS: Record<string, string> = {
  parse_and_clean: 'Parsing and expanding abbreviations',
  extract_entities: 'Extracting clinical entities',
  check_drug_interactions: 'Checking drug interactions',
  rag_enrich: 'Searching knowledge base',
  generate_summary: 'Generating final report',
}

const ALL_NODES = Object.keys(NODE_LABELS)

export default function AgentLog({ steps, status }: Props) {
  const completedNodes = new Set(steps.map(s => s.node))
  const lastCompleted = steps[steps.length - 1]?.node
  const lastCompletedIndex = ALL_NODES.indexOf(lastCompleted)
  const activeIndex = status === 'running' ? lastCompletedIndex + 1 : -1

  return (
    <div className="mt-4 pt-4 border-t border-stone-100 dark:border-stone-800">
      <div className="text-xs font-medium text-stone-400 dark:text-stone-500 mb-3">
        Agent progress
      </div>

      {/* Progress bar */}
      <div className="h-1 bg-stone-100 dark:bg-stone-800 rounded-full mb-4 overflow-hidden">
        <div
          className="h-full bg-brand-500 rounded-full transition-all duration-500"
          style={{ width: `${(completedNodes.size / ALL_NODES.length) * 100}%` }}
        />
      </div>

      <div className="space-y-1.5">
        {ALL_NODES.map((node, i) => {
          const isDone = completedNodes.has(node)
          const isActive = i === activeIndex
          const isPending = !isDone && !isActive

          return (
            <div
              key={node}
              className={`flex items-center gap-2.5 py-1.5 px-2 rounded-lg transition-all duration-300 ${
                isDone ? 'animate-slide-in' : ''
              } ${isActive ? 'bg-brand-50 dark:bg-brand-950/20' : ''}`}
            >
              {isDone && (
                <CheckCircle size={14} className="text-green-500 flex-shrink-0" />
              )}
              {isActive && (
                <Loader size={14} className="text-brand-500 flex-shrink-0 animate-spin" />
              )}
              {isPending && (
                <Circle size={14} className="text-stone-300 dark:text-stone-600 flex-shrink-0" />
              )}
              <span className={`text-xs ${
                isDone ? 'text-stone-700 dark:text-stone-300' :
                isActive ? 'text-brand-600 dark:text-brand-400 font-medium' :
                'text-stone-400 dark:text-stone-500'
              }`}>
                {NODE_LABELS[node]}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}