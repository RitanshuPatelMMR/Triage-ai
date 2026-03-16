import { Component, ReactNode } from 'react'

interface Props { children: ReactNode }
interface State { hasError: boolean; error: string }

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: '' }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error: error.message }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-sm text-red-600">
          <p className="font-medium mb-1">Rendering error</p>
          <p className="text-xs">{this.state.error}</p>
          <button
            onClick={() => this.setState({ hasError: false, error: '' })}
            className="mt-2 text-xs px-3 py-1 border border-red-200 rounded-lg hover:bg-red-100"
          >
            Try again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
