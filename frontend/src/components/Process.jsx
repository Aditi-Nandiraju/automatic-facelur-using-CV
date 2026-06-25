import { useState } from 'react'

async function parseResponse(res) {
  const text = await res.text()
  let data
  try { data = JSON.parse(text) } catch { data = { detail: text } }
  if (!res.ok) throw new Error(data.detail || `Server error ${res.status}`)
  return data
}

const STAGES = ['Tracking forward…', 'Tracking backward…', 'Writing output…']

export default function Process({ videoInfo, selection }) {
  const { videoId, totalFrames, fps } = videoInfo
  const { frameIdx, box } = selection

  const [status,       setStatus]       = useState('idle')
  const [stage,        setStage]        = useState(0)
  const [error,        setError]        = useState(null)
  const [downloading,  setDownloading]  = useState(false)

  function startFakeProgress() {
    setStage(0)
    const t1 = setTimeout(() => setStage(1), (totalFrames / fps) * 400)
    const t2 = setTimeout(() => setStage(2), (totalFrames / fps) * 800)
    return () => { clearTimeout(t1); clearTimeout(t2) }
  }

  async function run() {
    setStatus('running')
    setError(null)
    const stop = startFakeProgress()
    try {
      const res  = await fetch('/api/process', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ videoId, frameIdx, box }),
      })
      stop()
      await parseResponse(res)   // throws with a readable message on any error
      setStatus('done')
    } catch (e) {
      stop()
      setError(e.message)
      setStatus('error')
    }
  }

  // Fetch the file as a blob and trigger a browser download.
  // Direct <a href> through a dev proxy can silently drop binary responses.
  async function download() {
    setDownloading(true)
    try {
      const res = await fetch(`/api/download/${videoId}`)
      if (!res.ok) {
        const text = await res.text()
        let msg; try { msg = JSON.parse(text).detail } catch { msg = text }
        throw new Error(msg || `Server error ${res.status}`)
      }
      const blob = await res.blob()
      const url  = URL.createObjectURL(blob)
      const a    = document.createElement('a')
      a.href     = url
      a.download = `blurred_${videoId.slice(0, 8)}.mp4`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (e) {
      setError(`Download failed: ${e.message}`)
    }
    setDownloading(false)
  }

  const progressPct = status === 'running' ? [33, 66, 90][stage] : status === 'done' ? 100 : 0

  return (
    <div>
      <div className="process-summary">
        Selected face at <strong>frame {frameIdx}</strong> ·{' '}
        box [{box.join(', ')}] ·{' '}
        {totalFrames} frames to process
      </div>

      {status !== 'done' && (
        <button className="btn btn-primary" onClick={run} disabled={status === 'running'}>
          {status === 'running' ? 'Processing…' : 'Blur this face'}
        </button>
      )}

      {status === 'running' && (
        <>
          <div className="progress-bar">
            <div className="progress-bar-fill" style={{ width: `${progressPct}%` }} />
          </div>
          <div className="progress-label">{STAGES[stage]}</div>
        </>
      )}

      {error && <div className="error-box">{error}</div>}

      {status === 'done' && (
        <div className="download-row">
          <span className="done-badge">✓ Done</span>
          <button className="btn btn-success" onClick={download} disabled={downloading}>
            {downloading ? 'Preparing…' : 'Download blurred video'}
          </button>
          <button className="btn btn-primary" onClick={() => { setStatus('idle'); setError(null) }}>
            Re-process
          </button>
        </div>
      )}
    </div>
  )
}
