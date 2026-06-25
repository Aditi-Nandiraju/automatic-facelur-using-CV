import { useState } from 'react'
import Upload from './components/Upload'
import Scrubber from './components/Scrubber'
import Process from './components/Process'

export default function App() {
  const [videoInfo, setVideoInfo]   = useState(null)
  const [videoFile, setVideoFile]   = useState(null)
  const [selection, setSelection]   = useState(null)

  function handleUpload({ info, file }) {
    setVideoInfo(info)
    setVideoFile(file)
    setSelection(null)
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Face Blur</h1>
        <p>Upload a video, pick the face, download the result.</p>
      </header>

      <main className="app-main">
        <Step n={1} label="Upload your video">
          <Upload onUpload={handleUpload} />
        </Step>

        {videoInfo && videoFile && (
          <Step n={2} label="Scrub to a clear frame and click the face to blur">
            <Scrubber videoInfo={videoInfo} videoFile={videoFile} onSelect={setSelection} />
          </Step>
        )}

        {selection && (
          <Step n={3} label="Process and download">
            <Process videoInfo={videoInfo} selection={selection} />
          </Step>
        )}
      </main>
    </div>
  )
}

function Step({ n, label, children }) {
  return (
    <section className="step">
      <div className="step-heading">
        <span className="step-num">{n}</span>
        <span className="step-label">{label}</span>
      </div>
      <div className="step-body">{children}</div>
    </section>
  )
}
